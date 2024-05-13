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
from botocore.exceptions import ClientError
from botocore.exceptions import BotoCoreError
import gzip
import csv



logger = logging.getLogger()
logger.setLevel(logging.INFO)

openai.api_key = os.environ['OPENAI_API']

bedrock_agent_runtime = boto3.client(
    service_name = "bedrock-agent-runtime"
)

    
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

def retrieve(query):
    modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
    # modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-opus-20240229-v1:0'
    kbId = "RQ7PKC2IZP"
    
    prompt = f"""
    You are an AI assistant created by Anthropic to be helpful, harmless, and honest. The user has provided the following question:

    - Question: {query}

    Please provide a thoughtful and informative response to the user's question. 
    If you do not have enough information to provide a complete answer, 
    please indicate that you are unable to fully address the query and suggest ways the user could provide more details to help you assist them better. Answer in Korean.
    """
    
    try:
        response = bedrock_agent_runtime.retrieve_and_generate(
            input={
                'text': prompt,
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': kbId,
                    'modelArn': modelArn
                }
            }
        )
            
        review_result = response['output']['text'].strip().lower()
        logger.info('### retrieve response: %s', review_result)
        if 'sorry' in review_result or '죄송합니다' in review_result:
            response['output']['text'] = '⚠️ 죄송해요, 해당 요청은 처리할 수 없어요. 조금 더 구체적인 정보를 제공해주시거나 다른 질문을 해주시면 도움을 드릴 수 있을 것 같아요.'

        return response
    except Exception as e:
        logger.error(e)
        return {'error': '예기치 않은 오류가 발생했습니다.', 'details': str(e)}

    
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
            if len(source_text) > 15:
                source_text = source_text[:15] + '...'
            
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


def get_suggestion_from_metadata_os_picker(intent_request, session_attributes):
    logger.info('########## get_suggestion_from_metadata_os')

    try:
        modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
        kbId = "RQ7PKC2IZP"

        prompt = f"""
        Recommend questions users can ask you based on your knowledge base. No source information is included. Answer in Korean.
        """

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

        logger.info('get_suggestion_from_metadata_os-response: %s', response)

        question_list = response['output']['text'].split('\n')[1:]  # 첫 번째 요소 제외

        list_picker_content = {
            "templateType": "ListPicker",
            "version": "1.0",
            "data": {
                "replyMessage": {
                    "title": "질문 선택해주세요.",
                    "subtitle": "아래 질문 중에서 선택하세요.",
                    "imageType": "URL",
                    "imageData": "https://interactive-msg.s3-us-west-2.amazonaws.com/fruit_34.3kb.jpg",
                    "imageDescription": "질문 선택하기"
                },
                "content": {
                    "title": "사용자가 물어볼 수 있는 질문",
                    "subtitle": "질문을 선택해주세요.",
                    "imageType": "URL",
                    "imageData": "https://interactive-msg.s3-us-west-2.amazonaws.com/fruit_34.3kb.jpg",
                    "imageDescription": "질문 선택하기",
                    "elements": [
                        {
                            "title": question,
                            "subtitle": "",
                            "imageType": "URL",
                            "imageData": "https://interactive-message-testing.s3-us-west-2.amazonaws.com/apple_4.2kb.jpg"
                        } for question in question_list
                    ]
                }
            }
        }

        session_attributes['appContext'] = json.dumps(list_picker_content)

        return build_response(
            intent_request=intent_request,
            session_attributes=session_attributes,
            fulfillment_state="Fulfilled",
            message={
                'contentType': 'PlainText',
                'content': '질문을 선택해주세요.'
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


def get_suggestion_from_metadata_os(intent_request, session_attributes):
    logger.info('################ get_suggestion_from_metadata_os ################')
    try:
        modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
        # modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-opus-20240229-v1:0'
        # kbId = "JS9ZJONAQY"  # readme.txt 
        kbId = "RQ7PKC2IZP"
        
        
        # prompt = f"""
        # 당신이 갖고있는 지식 기반으로 사용자가 당신에게 물어볼 수 있는 질문을 추천해주세요.
        # """
        
        prompt = f"""
        Recommend questions users can ask you based on your knowledge base. No source information is included. Answer in Korean.
        """
    
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
        
        content = response['output']['text'] + '<br><br>📚 학습 정보: <a href="https://www.kbsec.com/go.able?linkcd=m06100004">KB증권 홈페이지 약관/유의사항</a>'
        logger.info('get_suggestion_from_metadata_os-response: %s', response)
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
       

def get_suggestion_from_metadata_os_btn(intent_request, session_attributes):
    logger.info('########## get_suggestion_from_metadata_os')
    
    try:
        modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
        # modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-opus-20240229-v1:0'
        # kbId = "JS9ZJONAQY"  # readme.txt 
        kbId = "RQ7PKC2IZP"
        
        
        # prompt = f"""
        # 당신이 갖고있는 지식 기반으로 사용자가 당신에게 물어볼 수 있는 질문을 추천해주세요.
        # """
        
        prompt = f"""
        Recommend questions users can ask you based on your knowledge base. No source information is included. Answer in Korean.
        """
    
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
        
        logger.info('get_suggestion_from_metadata_os-response: %s', response)
        
        question_list = response['output']['text'].split('\n')
        buttons = []
        for question in question_list[1:]:  # 첫 번째 요소는 제외
            if question.strip():
                buttons.append({
                    'text': question.strip().replace('- ', ''),
                    'value': question.strip().replace('- ', '')
                })
        
        content = '사용자가 질문할 수 있는 예시입니다. 버튼을 클릭해 보세요:<br><br>'
        content += '📚 학습 정보: <a href="https://www.kbsec.com/go.able?linkcd=m06100004">KB증권 홈페이지 약관/유의사항</a>'
        
        logger.info('########## question_list: %s', question_list)
        logger.info('########## question: %s', question)
        logger.info('########## buttons: %s', buttons)
        
        app_context = {
            "altMessages": {
                "markdown": content
            },
            "buttons": buttons
        }
        
        session_attributes['appContext'] = json.dumps(app_context)
        logger.info('########## appContext: %s', session_attributes['appContext'])
        
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
       

def review_query_with_metadata_os(query):
    try:
        modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
        # modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-opus-20240229-v1:0'
        # kbId = "JS9ZJONAQY"  # readme.txt 
        kbId = "RQ7PKC2IZP"
        
        # TODO Shuan metadata 자동화 ######
        # AS-IS: 지금은 수동으로 readme 파일을 등록해놨음 - S3:1, OS:2
        # s3://kb-able-talk-s3/readme.txt 
        # [lambda] metadata 파일 읽고 > [lambda] csv.zp 파일 파싱 to text > [lambda] S3 readme.txt 업데이트 > S3 이벤트 트리거
        # [lambda] readme 파일 읽고   > [lambda] 임베딩 (JS9ZJONAQY)
        # prompt에 metadata를 녹이려고 했으나, 제한된 INPUT length로 metadata 정보를 바라보는 OS를 별도로 둠
        # 개선사항: api를 호출할때 document를 첨부할 수 있거나, metadata url을 참조할 수 있게 프롬프팅이 된다면- 별도의 임베딩과 OS는 필요없음 ######
        
        # prompt = f"""
        # 질문: "{query}"
        # 이 질문에 대한 답변을 제공할 수 있는지 검토해 주세요. 가능하다면 관련 정보를 기반으로 답변을 준비하고, 불가능할 경우 '죄송합니다'라고 응답해 주세요.
        # """
        
        prompt = f"""
        question: "{query}"
        Please consider if you can provide an answer to this question. If possible, prepare a response based on relevant information, and if not possible, respond with 'sorry'.
        No source information is included.
        """
    
        logger.info('prompt: %s', prompt)
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
        
        review_result = response['output']['text'].strip().lower()
        logger.info('review_result response: %s', review_result)
        # todo shuan
        # [INFO]	2024-04-26T01:27:34.694Z	3841859b-6487-4e7b-ba3d-1c88c3c473d0	review_result response: the search results do not contain specific information about financial product fees and costs. the results mostly contain various financial service agreements and terms of use, but do not provide a comprehensive overview of fees and costs associated with different financial products. without more detailed information, i cannot provide a satisfactory answer to the question about financial product fees and costs.

        
        # if 'sorry' in review_result or 'do not contain' in review_result or '죄송합니다' in review_result or '검색 결과' in review_result or '찾을 수 없습니다' in review_result:
        if any(phrase in review_result for phrase in ['sorry', 'do not contain', '죄송합니다', '검색 결과', '찾을 수 없습니다']):
            logger.info('[review_query_with_metadata_os] can_answer: False')
            return {'can_answer': False}
        else:
            return {'can_answer': True}
    except Exception as e:
        logger.error('Exception: %s', {str(e)})
        # return {'can_answer': False, 'message': f'질문 검토 중 오류가 발생했습니다: {str(e)}'}
        return {'can_answer': False}
    
def handle_rag(intent_request, content_data, session_attributes):
    try:
        source_text = None
        source_location = None
        
        if not content_data:
            return elicit_slot(
                session_attributes=session_attributes,
                intent_name='Reception',
                slots=get_slots(intent_request),
                slot_to_elicit='ContentData',
                message={
                    'contentType': 'PlainText',
                    'content': '⚠️ 이용약관 및 유의사항에 대한 질문을 입력해주세요.'
                }
            )
        
        doc_list = get_s3_inventory_data() ## TODO metadata 파싱 결과 - 아직 사용은 안함
        
        # 사용자 질문에 답변 가능한지 검토 요청
        review_response = review_query_with_metadata_os(content_data)
        
        if review_response['can_answer']:
            response = retrieve(content_data)
            logger.info('retrieve-response: %s', response)
            citations_data = extract_citation_data(response)
            content = format_response_with_citations(response['output']['text'], citations_data)
            
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
        else:
            return fallbackIntent(intent_request, content_data, session_attributes)
        
    except Exception as e:
        logger.error('Exception: %s', e, exc_info=True)
        return fallbackIntent(intent_request, content_data, session_attributes)

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
            return get_suggestion_from_metadata_os(intent_request, session_attributes)
        
        
        # '번역' 태스크에 대한 슬롯 값 가져오기
        # translation_language = get_slot(intent_request, 'TranslationLanguage') if task_type == '번역' else None
        
        # 각 태스크 유형별로 처리할 핸들러 함수를 사전에 매핑
        # task_handlers = {
        #     '요약': handle_summary,
        #     '번역': handle_translation,
        #     '문구생성': handle_gen_text,
        #     'sql': handle_gen_sql,
        #     '기타': handle_etc,
        #     '문서리뷰': handle_doc_summary,
        #     'rag': handle_rag
        # }
        
        # return task_handlers[task_type](intent_request, content, session_attributes)
        
        
        
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


