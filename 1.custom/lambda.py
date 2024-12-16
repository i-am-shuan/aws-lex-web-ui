import openai
import json
import random
import decimal
import os
import logging
import chardet
import PyPDF2
from PyPDF2 import PdfReader
from io import BytesIO
import boto3
from urllib.parse import unquote
from urllib.parse import urlparse
import botocore
from botocore.client import Config
from botocore.exceptions import ClientError
from botocore.exceptions import BotoCoreError
import gzip
import csv
import re
import pprint
import html
import asyncio
import time

pp = pprint.PrettyPrinter(indent=2)

region = 'us-east-1'
bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
bedrock_client = boto3.client('bedrock-runtime', region_name=region)
bedrock_agent_client = boto3.client("bedrock-agent-runtime", config=bedrock_config, region_name=region)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

bedrock_agent_runtime = boto3.client(service_name="bedrock-agent-runtime")


AWS_REGION = os.environ["AWS_REGION"]
ENDPOINT_URL = os.environ.get("ENDPOINT_URL", f'https://bedrock-runtime.{AWS_REGION}.amazonaws.com')
modelId = "anthropic.claude-3-haiku-20240307-v1:0"

accept = "application/json"
contentType = "application/json"

dynamodb_client = boto3.resource('dynamodb')
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=AWS_REGION, endpoint_url=ENDPOINT_URL)
s3 = boto3.client('s3')

##################################################################
def lambda_handler(event, context):
    try:
        logger.info('Event: %s', json.dumps(event))
        
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(dispatch(event))
        return response
    except Exception as e:
        return handle_exception(e, event, get_session_attributes(event))
        
async def dispatch(intent_request):
    intent_name = intent_request['sessionState']['intent']['name']
    content = get_slot(intent_request, 'ContentData')
    session_attributes = get_session_attributes(intent_request)
    
    return await Reception(intent_request)        

def router(event, context):
    intent_name = event['sessionState']['intent']['name']
    sess_id = event['sessionId']
    result = asyncio.run(openai_async_api_handler(event, context))
    print(result)
    return result

async def openai_async_api_handler(event, context):
    try:
        sessionId = event['sessionId']
        inputTranscript = event['inputTranscript']
        
        # JSON payload 구성
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": inputTranscript
                }
            ]
        })
        
        session_attributes = get_session_attributes(event)
        print('sessionId ', sessionId)
        print('inputTranscript ', inputTranscript)
        fullreply = ''
        
        if 'streamingDynamoDbTable' in session_attributes and 'streamingEndpoint' in session_attributes:
            apigatewaymanagementapi = boto3.client(
                'apigatewaymanagementapi', 
                endpoint_url=session_attributes['streamingEndpoint']
            )
            
            wstable = dynamodb_client.Table(session_attributes['streamingDynamoDbTable'])
            print('wstable: ', wstable)
            print('wstable-scan: ', wstable.scan())
            
            db_response = wstable.get_item(Key={'sessionId': sessionId})
            print('db_response: ', db_response)
            connectionId = db_response['Item']['connectionId']
            print('Get ConnectionID ', connectionId)
    
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=body, modelId=modelId, accept=accept, contentType=contentType
            )
            stream = response.get('body')
    
            if stream:
                for streamEvent in stream:
                    chunk = streamEvent.get('chunk')
                    if chunk:
                        try:
                            chunk_obj = json.loads(chunk.get('bytes').decode())
                            if 'delta' in chunk_obj and 'text' in chunk_obj['delta']:
                                text = chunk_obj['delta']['text']
                                fullreply += text
                                print(text)
                                apigatewaymanagementapi.post_to_connection(
                                    Data=text.encode('utf-8'),
                                    ConnectionId=connectionId
                                )
                        except botocore.exceptions.ClientError as e:
                            if e.response['Error']['Code'] == 'GoneException':
                                print(f"GoneException occurred for connectionId: {connectionId}")
                                # Handle the connection being gone, e.g., marking the connection as closed in your DB.
                            else:
                                raise e
        else:
            response = bedrock_runtime.invoke_model(
                body=body, modelId=modelId, accept=accept, contentType=contentType
            )
            response_body = json.loads(response["body"].read())
            print("Response Body: ", response_body)  # Response Body 로그 추가
            fullreply = response_body.get("completion", '')  # 안전하게 'completion' 필드 추출
        
        message = {
            'contentType': 'CustomPayload',
            'content': fullreply
        }
        fulfillment_state = "Fulfilled"
        return close(event, session_attributes, fulfillment_state, message)
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
        return {
            'statusCode': 500,
            'body': json.dumps('Internal Server Error')
        }

def update_session_ttl(session_id, new_ttl):
    try:
        table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'kb-able-talk-prod-CodeBuildDeploy-VFBRRXN1GMAF-streaming')  # 환경 변수에서 테이블 이름을 가져옴
        table = dynamodb_client.Table(table_name)
        
        # 항목 존재 여부 확인
        response = table.get_item(Key={'sessionId': session_id})
        if 'Item' not in response:
            logger.error(f"Session ID {session_id} not found in table {table_name}")
            return
        
        # TTL 업데이트
        response = table.update_item(
            Key={'sessionId': session_id},
            UpdateExpression='SET #ttl = :ttl',
            ExpressionAttributeNames={'#ttl': 'ttl'},
            ExpressionAttributeValues={':ttl': new_ttl}
        )
        logger.info(f"Updated session TTL for sessionId {session_id} to {new_ttl}")
    except ClientError as e:
        logger.error(f"Error updating TTL for sessionId {session_id}: {e}")



async def handle_rag(intent_request, query, session_attributes):
    try:
        retrieval_results = retrieve_rag(query)
        logger.info("@@retrieval_results: %s", retrieval_results)
        
        filtered_results = []
        if query != '사용 예시를 알려주세요.':
            min_score = 0.9
            filtered_results = [result for result in retrieval_results if result['score'] >= min_score]
        logger.info("@@filtered_results: %s", filtered_results)
        
        contexts = get_contexts(filtered_results)
        logger.info("@@contexts: %s", contexts)
        
        prompt = f"""
        Human: 질문을 받으면 다음의 메시지를 반환합니다. <br> 태그부분에는 개행이 되도록 해줘.
        [KB증권] 마케팅 정보 수신동의
        <br>
        고객님, 안녕하세요.

        고객님은 2018년 12월 5일에 KB증권 마케팅 정보 수신에 동의하셨어요. 계속 유지하시면 다양한 혜택과 정보를 받아볼 수 있어요.

        이 이메일은 정보통신망 이용촉진 및 정보보호 등에 관한 법률 제50조 제8항에 따라 고객님께 2년마다 마케팅 정보 수신동의 유지 여부를 확인하기 위해 보내드리고 있습니다.

        ■ 수신동의 철회방법 <br>
        ① KB증권 홈페이지: 뱅킹/대출 > 개인(신용)정보동의서 <br>
        ② H-able: 개인(신용)정보동의서(화면번호 #0829) <br>
        ③ M-able 앱: 고객서비스 > 개인(신용)정보동의서 <br>
        ④ 전화: KB증권 고객센터(1588-6611) <br>

        본 메일은 발신전용 메일입니다.
        문의사항은 KB증권 고객센터(1588-6611)로 연락주세요.

        KB증권
        서울특별시 영등포구 여의나루로 50 | 대표번호 : 1588-6611
        
        <question>
        {query}
        </question>
        
        The response should be specific and use statistics or numbers when possible.
        
        Assistant:"""

        sessionId = intent_request['sessionId']
        wstable = dynamodb_client.Table(session_attributes['streamingDynamoDbTable'])
        
        db_response = wstable.get_item(Key={'sessionId': sessionId})
        logger.info('db_response: %s', db_response)
        connection_id = db_response['Item']['connectionId']
        logger.info('Get ConnectionID %s', connection_id)

        apigatewaymanagementapi = boto3.client(
            'apigatewaymanagementapi', 
            endpoint_url=session_attributes['streamingEndpoint']
        )
        
        new_ttl = int(time.time()) + 3600
        update_session_ttl(sessionId, new_ttl)
        
        content = await invoke_claude3(prompt, connection_id, apigatewaymanagementapi)
        content += generate_accessible_s3_urls(filtered_results)
        logger.info("@@@@@@@@@@@@@@@@@@content: %s", content)

        app_context = {
            "altMessages": {
                "markdown": content
            }
        }
        
        session_attributes['appContext'] = json.dumps(app_context)
        
        return build_response(
            intent_request=intent_request,
            session_attributes=session_attributes,
            fulfillment_state="Fulfilled",
            message={
                'contentType': 'PlainText',
                'content': content
            }
        )
        
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
        return fallbackIntent(intent_request, query, session_attributes)


async def handle_rag2(intent_request, query, session_attributes):
    try:
        retrieval_results = retrieve_rag(query)
        logger.info("@@retrieval_results: %s", retrieval_results)
        
        filtered_results = []
        if query != '사용 예시를 알려주세요.':
            min_score = 0.9
            filtered_results = [result for result in retrieval_results if result['score'] >= min_score]
        logger.info("@@filtered_results: %s", filtered_results)
        
        contexts = get_contexts(filtered_results)
        logger.info("@@contexts: %s", contexts)
        
        prompt = f"""
        Human: You are a financial advisor AI system, and provides answers to questions by using fact based and statistical information when possible. 
        Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. And make an answer in Korean. 
        <context>
        {contexts}
        </context>
        
        <question>
        {query}
        </question>
        
        The response should be specific and use statistics or numbers when possible.
        
        Assistant:"""

        sessionId = intent_request['sessionId']
        wstable = dynamodb_client.Table(session_attributes['streamingDynamoDbTable'])
        
        db_response = wstable.get_item(Key={'sessionId': sessionId})
        logger.info('db_response: %s', db_response)
        connection_id = db_response['Item']['connectionId']
        logger.info('Get ConnectionID %s', connection_id)

        apigatewaymanagementapi = boto3.client(
            'apigatewaymanagementapi', 
            endpoint_url=session_attributes['streamingEndpoint']
        )
        
        new_ttl = int(time.time()) + 3600
        update_session_ttl(sessionId, new_ttl)
        
        content = await invoke_claude3(prompt, connection_id, apigatewaymanagementapi)
        content += generate_accessible_s3_urls(filtered_results)
        logger.info("@@@@@@@@@@@@@@@@@@content: %s", content)

        app_context = {
            "altMessages": {
                "markdown": content
            }
        }
        
        session_attributes['appContext'] = json.dumps(app_context)
        
        return build_response(
            intent_request=intent_request,
            session_attributes=session_attributes,
            fulfillment_state="Fulfilled",
            message={
                'contentType': 'PlainText',
                'content': content
            }
        )
        
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
        return fallbackIntent(intent_request, query, session_attributes)



def escape_special_chars(text):
    try:
        return html.escape(text)
    except Exception as e:
        logger.exception("Error escaping special characters: %s", str(e))
        return text  # 변환 실패 시 원본 텍스트 반환

async def invoke_claude3(prompt, connection_id, apigatewaymanagementapi):
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"

    try:
        response = await asyncio.to_thread(
            bedrock_client.invoke_model_with_response_stream,
            modelId=model_id,
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4096,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ],
                }
            ),
        )

        stream = response.get("body")
        full_reply = ""

        if stream:
            for stream_event in stream:
                chunk = stream_event.get("chunk")
                if chunk:
                    chunk_obj = json.loads(chunk.get("bytes").decode())
                    # logger.info("Chunk Object: %s", chunk_obj)
                    if 'delta' in chunk_obj and 'text' in chunk_obj['delta']:
                        text = chunk_obj['delta']['text']
                        full_reply += text
                        # logger.info("Chunk Text: %s", text)

                        # 실시간 스트리밍 데이터 전송
                        try:
                            apigatewaymanagementapi.post_to_connection(
                                Data=text.encode('utf-8'),
                                ConnectionId=connection_id
                            )
                        except botocore.exceptions.ClientError as e:
                            if e.response['Error']['Code'] == 'GoneException':
                                logger.error(f"GoneException occurred for connection_id: {connection_id}")
                                # 동기적으로 Claude 모델 호출
                                return invoke_claude3_sync(prompt)
                            else:
                                raise e

        return full_reply

    except ClientError as err:
        logger.error(
            "Couldn't invoke Claude 3 asynchronously. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise


def invoke_claude3_sync(prompt):
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"

    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4096,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ],
                }
            ),
        )

        result = json.loads(response.get("body").read())
        input_tokens = result["usage"]["input_tokens"]
        output_tokens = result["usage"]["output_tokens"]
        output_list = result.get("content", [])

        return '⚠️ 세션이 만료되었습니다. 🔄 브라우저를 새로고침 해주세요. <br><br>'+ output_list[0]["text"] if output_list else ""

    except ClientError as err:
        logger.error(
            "Couldn't invoke Claude 3 synchronously. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise


def extract_uris_and_text(retrieval_results):
    uris = []
    texts = []
    for result in retrieval_results:
        if 'location' in result and 's3Location' in result['location'] and 'uri' in result['location']['s3Location']:
            uri = result['location']['s3Location']['uri']
            uris.append(uri)
        if 'content' in result and 'text' in result['content']:
            text = result['content']['text']
            texts.append(text)
    return uris, texts
    

def generate_s3_url(source_location):
    try:
        s3 = boto3.client('s3')
        bucket_name, key = source_location.replace('s3://', '').split('/', 1)
        url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': bucket_name,
                'Key': key
            },
            ExpiresIn=3600
        )
        return url
    except ClientError as e:
        logger.error(e)
        return None

def generate_accessible_s3_urls(retrieval_results):
    uris, texts = extract_uris_and_text(retrieval_results)
    html_output = ""
    first_time = True 
    processed_files = set()

    for i, uri in enumerate(uris):
        url = generate_s3_url(uri)
        file_name = uri.split('/')[-1]

        if file_name not in processed_files:
            processed_files.add(file_name)
            
            if first_time:
                html_output += "<br><br>📖 <b>출처</b><br>"
                first_time = False

            escaped_text = html.escape(texts[i])
            if len(escaped_text) > 500:
                    logger.warning("Escaped text is too long, truncating to 500 characters.")
                    escaped_text = escaped_text[:500] + '...'
            html_output += f'<a href="{url}" target="_blank" title="{escaped_text}">{file_name}</a><br>'
            # html_output += f'<a href="{url}">{file_name}</a><br>'
    
    # print("@@@@@@escaped_text: ", escaped_text)
    return html_output
    

######################################################################

def get_session_attributes(intent_request):
    sessionState = intent_request['sessionState']
    if 'sessionAttributes' in sessionState:
        return sessionState['sessionAttributes']
    else:
        return {}

def get_slot(intent_request, slotName):
    slots = get_slots(intent_request)
    if slots is not None and slotName in slots:
        slot = slots[slotName]
        if slot is not None and 'value' in slot and slot['value'] is not None:
            return slot['value'].get('interpretedValue')
    return None

def get_slots(intent_request):
    return intent_request['sessionState']['intent']['slots']

def build_response(intent_request, session_attributes, fulfillment_state, message):
    return {
        'sessionState': {
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'Close'
            },
            'intent': {
                'name': intent_request['sessionState']['intent']['name'],
                'state': fulfillment_state
            }
        },
        'messages': [message] if message else [],
        'requestAttributes': intent_request['requestAttributes'] if 'requestAttributes' in intent_request else None
    }

def fallbackIntent(intent_request, content_data, session_attributes):
    try:
        # logger.info('fallbackIntent-content_data: %s', content_data)
        response = '⚠️ 세션이 만료되었습니다. 🔄 브라우저를 새로고침 해주세요.'
        
        app_context = {
            "altMessages": {
                "markdown": response
            }
        }
        session_attributes['appContext'] = json.dumps(app_context)
    
        return build_response(
            intent_request=intent_request,
            session_attributes=session_attributes,
            fulfillment_state="Fulfilled",
            message={
                'contentType': 'PlainText',
                'content': response
            }
        )
    except Exception as e:
        return build_response(
            intent_request=intent_request,
            session_attributes=session_attributes,
            fulfillment_state="Fulfilled",
            message={
                'contentType': 'PlainText',
                'content': str(e)
            }
        )

def retrieve_rag(query):
    try:
        numberOfResults = 5
        kbId = "RQ7PKC2IZP"
        
        relevant_documents = bedrock_agent_client.retrieve(
            retrievalQuery= {
                'text': query
            },
            knowledgeBaseId=kbId,
            retrievalConfiguration= {
                'vectorSearchConfiguration': {
                    'numberOfResults': numberOfResults,
                }
            }
        )
        
        return relevant_documents['retrievalResults']
    except Exception as e:
        logger.error(e)
        return {'error': '예기치 않은 오류가 발생했습니다.', 'details': str(e)}

def get_contexts(retrievalResults):
    contexts = []
    for retrievedResult in retrievalResults: 
        contexts.append(retrievedResult['content']['text'])
    return contexts

async def Reception(intent_request):
    try:
        logger.info('intent_request: %s', intent_request)
        session_attributes = get_session_attributes(intent_request)
        content = intent_request['inputTranscript']
        
        s3_url = 's3://kb-able-talk-s3/questions.html'
        public_url = convert_s3_url_to_public_url(s3_url)
        
        if content == '사용 예시를 알려주세요.':
            return await handle_example_request(intent_request, session_attributes, public_url)

        return await handle_rag(intent_request, content, session_attributes)
    
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        return fallbackIntent(intent_request, content, session_attributes)

async def handle_example_request(intent_request, session_attributes, public_url):
    content = '🔍 <a href="' + public_url + '"><b>질문예시</b></a>를 확인해보세요.<br>📚 KB증권 약관 관련 학습 정보가 궁금하다면 <b><a href="https://www.kbsec.com/go.able?linkcd=m06100004">여기</a></b>를 참고해주시기 바랍니다.<br>'
                        
    app_context = {
        "altMessages": {
            "markdown": content
        }
    }
    session_attributes['appContext'] = json.dumps(app_context)
    
    return build_response(
        intent_request=intent_request,
        session_attributes=session_attributes,
        fulfillment_state="Fulfilled",
        message={
            'contentType': 'PlainText',
            'content': content
        }
    )

def convert_s3_url_to_public_url(s3_url):
    # S3 URL 파싱
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc
    key = parsed_url.path.lstrip('/')

    # S3 클라이언트 생성
    s3 = boto3.client('s3')

    # S3 객체의 공개 URL 생성
    public_url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={
            'Bucket': bucket_name,
            'Key': key
        },
        ExpiresIn=3600  # 1시간 동안 유효한 URL
    )

    return public_url
    
def handle_exception(e, intent_request, session_attributes):
    """
    이 함수는 예외 상황이 발생했을 때 사용자에게 적절한 응답을 제공합니다.
    """
    logger.error('Exception: %s', e, exc_info=True)
    error_message = f'처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요. 오류 내용: {str(e)}'

    return build_response(
        intent_request=intent_request,
        session_attributes=session_attributes,
        fulfillment_state="Failed",
        message={
            'contentType': 'PlainText',
            'content': error_message
        }
    )

def get_session_attributes(intent_request):
    session_attributes = intent_request['sessionState'].get('sessionAttributes', {})
    session_attributes['streamingDynamoDbTable'] = 'kb-able-talk-prod-CodeBuildDeploy-VFBRRXN1GMAF-streaming'
    session_attributes['streamingEndpoint'] = 'https://oxtu03t1a0.execute-api.us-east-1.amazonaws.com/Prod/'
    
    print('Session Attributes:', session_attributes)
    return session_attributes

def close(intent_request, session_attributes, fulfillment_state, message):
    intent_request['sessionState']['intent']['state'] = fulfillment_state
    return {
        'sessionState': {
            'dialogAction': {
                'type': 'Close'
            },
            'intent': intent_request['sessionState']['intent']
        },
        'messages': [message],
        'sessionId': intent_request['sessionId'],
        'requestAttributes': intent_request['requestAttributes'] if 'requestAttributes' in intent_request else None
    }
