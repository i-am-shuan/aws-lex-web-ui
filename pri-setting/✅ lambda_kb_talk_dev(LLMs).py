import openai
import json
import random
import decimal 
import os
import json
import random
import decimal 
import logging
import chardet
import PyPDF2
from PyPDF2 import PdfReader
from io import BytesIO
import boto3
from urllib.parse import unquote
from botocore.client import Config
from botocore.exceptions import ClientError
from botocore.exceptions import BotoCoreError
import gzip
import csv
import re
import pprint
import html
# from langchain.llms.bedrock import Bedrock                               #'typing_extensions' (unknown location)
# from langchain.retrievers.bedrock import AmazonKnowledgeBasesRetriever   #'typing_extensions' (unknown location)

pp = pprint.PrettyPrinter(indent=2)


region = 'us-east-1'
bedrock_client = boto3.client('bedrock-runtime', region_name = region)
bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
bedrock_agent_client = boto3.client("bedrock-agent-runtime", config=bedrock_config, region_name = region)

########################################################################################
logger = logging.getLogger()
logger.setLevel(logging.INFO)

bedrock_agent_runtime = boto3.client(service_name = "bedrock-agent-runtime")
########################################################################################

def elicit_intent(intent_request, session_attributes, message):
    return {
        'sessionState': {
            'dialogAction': {
                'type': 'ElicitIntent'
            },
            'sessionAttributes': session_attributes
        },
        'messages': [ message ] if message != None else None,
        'requestAttributes': intent_request['requestAttributes'] if 'requestAttributes' in intent_request else None
    }

def elicit_slot(session_attributes, intent_name, slots, slot_to_elicit, message):
    return {
        'sessionState': {
            'dialogAction': {
                'type': 'ElicitSlot',
                'slotToElicit': slot_to_elicit,
                'intentName': intent_name
            },
            'intent': {
                'name': intent_name,
                'slots': slots,
                'state': 'InProgress'
            },
            'sessionAttributes': session_attributes
        },
        'messages': [message] if message else []
    }

def close(intent_request, session_attributes, fulfillment_state, message):
    intent_request['sessionState']['intent']['state'] = fulfillment_state
    return {
        'sessionState': {
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'Close'
            },
            'intent': intent_request['sessionState']['intent']
        },
        'messages': [message],
        'sessionId': intent_request['sessionId'],
        'requestAttributes': intent_request['requestAttributes'] if 'requestAttributes' in intent_request else None
    }
    
def get_session_attributes(intent_request):
    sessionState = intent_request['sessionState']
    if 'sessionAttributes' in sessionState:
        return sessionState['sessionAttributes']
    else:
        return {}

def get_slots(intent_request):
    # 'sessionState'의 'intent'에서 'slots' 정보를 추출합니다.
    return intent_request['sessionState']['intent']['slots']
    
def get_slot(intent_request, slotName):
    # 'slots' 정보를 가져옵니다.
    slots = get_slots(intent_request)
    
    # 'slots'가 None이 아니고, 'slotName'이 'slots' 안에 있으며, 해당 슬롯이 None이 아닌 경우,
    # 'interpretedValue'를 반환합니다.
    if slots is not None and slotName in slots:
        slot = slots[slotName]
        # 슬롯의 값이 None이 아니고, 'value' 키가 있으며, 'value'가 None이 아닌 경우 'interpretedValue'를 반환합니다.
        if slot is not None and 'value' in slot and slot['value'] is not None:
            return slot['value'].get('interpretedValue')
    # 위 조건에 맞지 않는 경우, None을 반환합니다.
    return None

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
########################################################################################
def retrieve_rag(query):
    try:
        numberOfResults=5
        kbId = "RQ7PKC2IZP"     # kb-able-talk-s3
        
        relevant_documents = bedrock_agent_client.retrieve(
            retrievalQuery= {
                'text': query
            },
            knowledgeBaseId=kbId,
            retrievalConfiguration= {
                'vectorSearchConfiguration': {
                    'numberOfResults': numberOfResults,
                    # 'overrideSearchType': "HYBRID", # optional
                }
            }
        )
        
        print('@@@@@@@@@@@@@@relevant_documents: ', relevant_documents)
        
        return relevant_documents['retrievalResults']
    except Exception as e:
        logger.error(e)
        return {'error': '예기치 않은 오류가 발생했습니다.', 'details': str(e)}
    
def invoke_mistral_7b(prompt):
    # https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeMistral7B_section.html
    try:
        instruction = f"<s>[INST] {prompt} [/INST]"
        body = json.dumps({
            "prompt": instruction,
            "max_tokens":4096,
            "temperature":0.5,
            "top_k":50,
            "top_p":0.9
        })
        
        model_id = 'mistral.mistral-7b-instruct-v0:2'
        accept = 'application/json'
        contentType = 'application/json'
        response = bedrock_client.invoke_model(body=body, modelId=model_id, accept=accept, contentType=contentType)
        response_body = json.loads(response["body"].read())
        content = response_body.get('outputs')[0]['text']
        
        return content
    except Exception as e:
        logger.error(e)
        return {'error': '예기치 않은 오류가 발생했습니다.', 'details': str(e)}

def invoke_mixtral_8x7b(prompt):
    # https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeMixtral8x7B_section.html
    try:
        instruction = f"<s>[INST] {prompt} [/INST]"
        body = json.dumps({
            "prompt": instruction,
            "max_tokens":200,
            "temperature":0.5,
            "top_k":50,
            "top_p":0.9
        })
        
        model_id = "mistral.mixtral-8x7b-instruct-v0:1"
        accept = 'application/json'
        contentType = 'application/json'
        response = bedrock_client.invoke_model(body=body, modelId=model_id, accept=accept, contentType=contentType)
        response_body = json.loads(response["body"].read())
        content = response_body.get('outputs')[0]['text']
        
        return content
    except Exception as e:
        logger.error(e)
        return {'error': '예기치 않은 오류가 발생했습니다.', 'details': str(e)}


def invoke_llama3_8b(prompt):
    # https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_Llama3_InvokeLlama_section.html
    try:
        llama_prompt = f"""
        <|begin_of_text|>
        <|start_header_id|>user<|end_header_id|>
        {prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
        
        body = {
            "prompt": llama_prompt,
            "temperature": 0.5,
            "top_p": 0.9,
            "max_gen_len": 2048
        }

        response = bedrock_client.invoke_model(body=json.dumps(body), modelId="meta.llama3-8b-instruct-v1:0")
        response_body = json.loads(response["body"].read())
        print("#######response_body: ", response_body)
        
        response_text = response_body["generation"]
        response_text = response_text.lower()
        print("#######response_text: ", response_text)

        return response_text

    except ClientError:
        logger.error("Couldn't invoke Llama 3")
        raise

# todo Shuan
def invoke_claude3_stream(prompt):
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    prompt = body.get('prompt')
    
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    }
    
    request_payload = json.dumps(native_request)
    
    try:
        streaming_response = client.invoke_model_with_response_stream(
            modelId=model_id,
            contentType="application/json",
            body=request_payload
        )
        
        result = ""
        for event in streaming_response['body']:
            chunk = json.loads(event['chunk']['bytes'])
            print(chunk)
            if chunk.get("type") == "content.block_delta":
                result += chunk['delta'].get("text", "")
    
    
    



def invoke_claude3(prompt):
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"

    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    # "max_tokens": 1024,
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

        # Process and print the response
        result = json.loads(response.get("body").read())
        input_tokens = result["usage"]["input_tokens"]
        output_tokens = result["usage"]["output_tokens"]
        output_list = result.get("content", [])

        return output_list[0]["text"] if output_list else ""

    except ClientError as err:
        logger.error(
            "Couldn't invoke Claude 3 Sonnet. Here's why: %s: %s",
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
        # S3 클라이언트 생성
        s3 = boto3.client('s3')

        # source_location에서 버킷 이름과 키(파일 경로) 추출
        bucket_name, key = source_location.replace('s3://', '').split('/', 1)

        # 임시 URL 생성
        url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': bucket_name,
                'Key': key
            },
            ExpiresIn=3600  # 유효 기간(초)
        )

        return url
    except ClientError as e:
        logger.error(e)
        return None

def generate_accessible_s3_urls_back(retrieval_results):
    uris, texts = extract_uris_and_text(retrieval_results)
    accessible_urls = []
    html_output = ""
    first_time = True 
    processed_files = set()

    for i, uri in enumerate(uris):
        url = generate_s3_url(uri)
        file_name = uri.split('/')[-1]

        # 이미 처리된 파일명 SKIP
        if file_name not in processed_files:
            processed_files.add(file_name)
            accessible_urls.append(url)
            
            if first_time:
                html_output += "<br><br>📚 <b>출처</b><br>"
                first_time = False  # 이후 실행에서는 이 부분이 실행되지 않도록 플래그 변경
            
            # html_output += f'<a href="{url}" target="_blank">{file_name}</a><br>'
            html_output += f'<a href="{url}" target="_blank" title="{texts[i]}">{file_name}</a><br>'     # 마우스 오버 # output 길이제한 이슈
            

    # for uri in uris:
    #     url = generate_s3_url(uri)
    #     accessible_urls.append(url)
    #     file_name = uri.split('/')[-1]
    #     # 이미 처리된 파일명 SKIP
    #     if file_name not in processed_files:
    #     accessible_urls.append(url)
    #     processed_files.add(file_name) # 파일명을 집합에 추가
    #     html_output += f'<a href="{url}" target="\_blank">{file_name}</a><br>'


    return html_output

def generate_accessible_s3_urls(retrieval_results):
    uris, texts = extract_uris_and_text(retrieval_results)
    html_output = ""
    first_time = True 
    processed_files = set()

    for i, uri in enumerate(uris):
        url = generate_s3_url(uri)
        file_name = uri.split('/')[-1]

        # 이미 처리된 파일명 SKIP
        if file_name not in processed_files:
            processed_files.add(file_name)
            
            if first_time:
                html_output += "<br><br>📚 <b>출처</b><br>"
                first_time = False  # 이후 실행에서는 이 부분이 실행되지 않도록 플래그 변경

            # HTML 특수 문자를 이스케이프 처리
            escaped_text = html.escape(texts[i])
            html_output += f'<a href="{url}" target="_blank" title="{escaped_text}">{file_name}</a><br>'
    
    return html_output

def format_response_with_citations(text_response, citations_data):
    try:
        logger.info('citations_data: %s', citations_data)
        
        # 출처 정보가 없을 경우, 기존 응답을 그대로 반환합니다.
        if not citations_data:
            return str(text_response)
        
        # 중복된 (source_text, source_location) 세트 제거
        unique_citations = list(set(citations_data))
        
        # 출처 정보를 포함하는 문자열을 생성합니다.
        content = str(text_response) + '<br><br> 📚 <b>출처</b>'
        num_citations = min(len(unique_citations), 3)  # 최대 3개의 출처만 가져옴
        
        for i, (source_text, source_location) in enumerate(unique_citations[:3]):  # 최대 3개의 출처만 처리
            # 's3://kb-able-talk-s3/test/test.pdf' 형태에서 'test.pdf' 추출
            file_name = source_location.split('/')[-1]
            
            # 출처 텍스트가 00자 이상인 경우 00자까지만 표시하고 '...'을 추가
            # if len(source_text) > 15:
            #     source_text = source_text[:15] + '...'
            
            s3_url = generate_s3_url(source_location)
            
            # 링크 생성
            citation_string = '<br><b>[{}]</b> <a href="{}" target="\\\\_blank">{}</a> <br>({})'.format(i+1, s3_url, file_name, source_text)
            
            # 마지막 출처인 경우에는 쉼표를 추가하지 않습니다.
            if i < num_citations - 1:
                citation_string += ', '
            
            content += citation_string
        
        return content
    
    except Exception as e:
        logger.error('Error in format_response_with_citations: %s', str(e))
        return str(text_response)  # 예외 발생 시 기존 응답을 그대로 반환
    

def extract_citation_data(response):
    citations_data = []

    # 인용 목록에서 각 인용을 반복 처리합니다.
    if 'citations' in response and response['citations']:
        for citation in response['citations']:
            # 각 인용에서 추출된 참조를 반복 처리합니다.
            if 'retrievedReferences' in citation:
                for reference in citation['retrievedReferences']:
                    source_text = None
                    source_location = None
                    
                    # 출처 텍스트를 확인하고 변수에 저장합니다.
                    if 'content' in reference and 'text' in reference['content']:
                        source_text = reference['content']['text']
                        
                    # 출처가 있는 S3 위치를 확인하고 변수에 저장합니다.
                    if 'location' in reference and 's3Location' in reference['location']:
                        source_location = reference['location']['s3Location'].get('uri', '')
                        
                    if source_location and source_text:
                        citations_data.append((source_text, source_location))
                        
    logger.info('extract_citation_data: %s', citations_data)
    return citations_data


def parse_documents(csv_content):
    reader = csv.reader(csv_content.strip().split('\n'))
    document_list = []
    for row in reader:
        if len(row) == 2:
            bucket, path = row
            file_name = path.split('/')[-1].rsplit('.', 1)[0]
            document_list.append(file_name)
    return ", ".join(document_list)

    
def get_s3_inventory_data():
    # S3 클라이언트 생성
    s3 = boto3.client('s3')

    # 인벤토리 manifest 파일 읽기
    manifest_obj = s3.get_object(Bucket='kb-able-talk-s3', Key='metadata/kb-able-talk-s3/metadata/2024-04-23T01-00Z/manifest.json')
    manifest = json.loads(manifest_obj['Body'].read())
    
    # 인벤토리 CSV 파일 정보 추출
    inventory_file = manifest['files'][0]
    inventory_bucket = manifest['sourceBucket']
    inventory_key = inventory_file['key']
    
    # 인벤토리 CSV 파일 읽기
    inventory_obj = s3.get_object(Bucket=inventory_bucket, Key=inventory_key)
    inventory_content = gzip.decompress(inventory_obj['Body'].read()).decode('utf-8')
    
    # 인벤토리 CSV 파일 내용 디코딩
    decoded_content = '\n'.join([','.join([unquote(item) for item in line.split(',')]) for line in inventory_content.split('\n')])
    parsed_content = parse_documents(decoded_content)
    
    return parsed_content


def get_contexts(retrievalResults):
    contexts = []
    for retrievedResult in retrievalResults: 
        contexts.append(retrievedResult['content']['text'])
    return contexts

def retrieve_qa(intent_request, session_attributes):
    try:
        modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
        kbId = "RQ7PKC2IZP" # kb-able-talk-s3
        
        
        # prompt = f"""
        # Recommend questions users can ask you based on your knowledge base. To ensure an accurate answer, please be specific with your question.
        # """
        
        # query = 'Based on your knowledge base, come up with 10 questions that users might commonly ask you. Be specific with your questions.'
        query = 'Recommend questions users can ask you based on your knowledge base. To ensure an accurate answer, please be specific with your question.'
        
        prompt = f"""
        Human: You are a financial advisor AI system, and provides answers to questions by using fact based and statistical information when possible. 
        Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. Answers should be provided in Korean.
        
        <question>
        {query}
        </question>
        
        Assistant:"""
    
        response = bedrock_agent_runtime.retrieve_and_generate(
            input={
                'text': prompt
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': kbId,
                    'modelArn': modelArn
                }
            }
        )
        
        content = response['output']['text'] + '<br><br><a href="https://www.kbsec.com/go.able?linkcd=m06100004">📚 학습 정보</a>'
        
        logger.info('retrieve_qa: %s', response)
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
        return build_response(
            intent_request=intent_request,
            session_attributes=session_attributes,
            fulfillment_state="Fulfilled",
            message={
                'contentType': 'PlainText',
                'content': str(e)
            }
        )
       

def handle_rag(intent_request, query, session_attributes):
    try:
        retrieval_results = retrieve_rag(query)
        print("@@retrieval_results: ", retrieval_results)
        
        min_score = 0.6  # 최소 점수 임계값 설정
        filtered_results = [result for result in retrieval_results if result['score'] >= min_score]
        print("@@filtered_results: ", filtered_results)
        
        contexts = get_contexts(filtered_results)
        print("@@contexts: ", contexts)
        
        ###### prompt (S) ######
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
        ###### prompt (E) ######
        
        
        ###### llm model call (S) ######
        # https://docs.aws.amazon.com/bedrock/latest/userguide/service_code_examples_bedrock-runtime_invoke_model_examples.html
        
        # content = invoke_mistral_7b(prompt) + generate_accessible_s3_urls(filtered_results)
        # content = invoke_mixtral_8x7b(prompt) + generate_accessible_s3_urls(filtered_results)
        # content = invoke_llama3_8b(prompt) + generate_accessible_s3_urls(filtered_results)
        # content = invoke_claude3(prompt) + generate_accessible_s3_urls(filtered_results)
        content = invoke_claude3_stream(prompt)    # todo Shuan test
        
        print("@@@@@@@@@@@@@@@@@@content: ", content)
        
        ###### llm model call (E) ######

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






def fallbackIntent(intent_request, content_data, session_attributes):
    try:
        logger.info('fallbackIntent-content_data: %s', content_data)
        response = retrieve(content_data)
        
        app_context = {
            "altMessages": {
                "markdown": response['output']['text']
            }
        }
        session_attributes['appContext'] = json.dumps(app_context)
    
        return build_response(
            intent_request=intent_request,
            session_attributes=session_attributes,
            fulfillment_state="Fulfilled",
            message={
                'contentType': 'PlainText',
                'content': response['output']['text']
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

def handle_exception(e, intent_request, session_attributes):
    """
    이 함수는 예외 상황이 발생했을 때 사용자에게 적절한 응답을 제공합니다.
    """
    logger.error('Exception: %s', e, exc_info=True)
    error_message = f'처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요. 오류 내용: {str(e)}'

    # 오류 메시지로 사용자에게 응답을 보냅니다.
    return build_response(
        intent_request=intent_request,
        session_attributes=session_attributes,
        fulfillment_state="Failed",
        message={
            'contentType': 'PlainText',
            'content': error_message
        }
    )

def Reception(intent_request):
    try:
        logger.info('intent_request: %s', intent_request)
        session_attributes = get_session_attributes(intent_request)
        content = intent_request['inputTranscript']
        
        if content == '사용 예시를 알려주세요.':
            return retrieve_qa(intent_request, session_attributes)
        
        return handle_rag(intent_request, content, session_attributes)
    
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        return fallbackIntent(intent_request, content, session_attributes)


def dispatch(intent_request):
    intent_name = intent_request['sessionState']['intent']['name']
    content = get_slot(intent_request, 'ContentData')
    session_attributes = get_session_attributes(intent_request)
    
    return Reception(intent_request)

def lambda_handler(event, context):
    try:
        logger.info('Event: %s', json.dumps(event))
        
        response = dispatch(event)
        return response
    except Exception as e:
        return handle_exception(e, event, get_session_attributes(event))
    
