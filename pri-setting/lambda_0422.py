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
from botocore.exceptions import ClientError
from botocore.exceptions import BotoCoreError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

openai.api_key = os.environ['OPENAI_API']

s3 = boto3.client('s3')

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

# 사실 기반: Haiku, 창의적 생성: Sonnet
def retrieve(query, opt, intent_request, session_attributes):
    modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
    # kbId = "YYBSKXHSED"
    kbId = "RQ7PKC2IZP"
    
    
    try:
        if opt == 'sonnet':
            modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
        
        # Bedrock의 retrieve_and_generate 메소드를 호출합니다.
        response = bedrock_agent_runtime.retrieve_and_generate(
            input={
                'text': query,
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': kbId,
                    'modelArn': modelArn,
                }
            }
        )
            
        if 'output' in response and 'text' in response['output']:
            if 'Sorry, I am unable to assist you with this request' in response['output']['text']:
                logger.error("Bedrock retrieve_and_generate returned an error message.")
                response['output']['text'] = '🤖 죄송해요, 해당 요청은 처리할 수 없어요. 인터넷을 사용하는 실시간 정보 처리는 제가 할 수 없습니다. 좀 더 구체적인 정보를 제공해주시거나 다른 질문을 해주시면 도움을 드릴 수 있을 것 같아요.'
        
        
        return response
    except ClientError as e:
        logger.error(f"Error calling Bedrock retrieve_and_generate: {e}")
        return {'error': '서버 통신 중 문제가 발생했습니다.', 'details': str(e)}
    except BotoCoreError as e:
        logger.error(f"BotoCoreError calling Bedrock retrieve_and_generate: {e}")
        return {'error': 'AWS 서비스 호출에 문제가 발생했습니다.', 'details': str(e)}
    except Exception as e:
        logger.error(f"Unexpected error calling Bedrock retrieve_and_generate: {e}")
        return {'error': '예기치 않은 오류가 발생했습니다.', 'details': str(e)}


def extract_citation_data(response):
    source_text = None
    source_location = None

    # 'citations' 키의 존재 여부를 확인하고, 관련 데이터를 추출합니다.
    if 'citations' in response and response['citations']:
        first_citation = response['citations'][0]

        # 'retrievedReferences'의 존재 여부를 확인하고, 출처 정보를 가져옵니다.
        if 'retrievedReferences' in first_citation and first_citation['retrievedReferences']:
            first_reference = first_citation['retrievedReferences'][0]

            # 출처 텍스트가 있는지 확인하고 변수에 저장합니다.
            if 'content' in first_reference and 'text' in first_reference['content']:
                source_text = first_reference['content']['text']

            # 출처가 있는 S3 위치를 확인하고 변수에 저장합니다.
            if 'location' in first_reference and 's3Location' in first_reference['location']:
                source_location = first_reference['location']['s3Location'].get('uri', '')

    return source_text, source_location

def handle_rag(intent_request, content_data, session_attributes):
    if not content_data:
        return elicit_slot(
            session_attributes=session_attributes,
            intent_name='Reception',
            slots=get_slots(intent_request),
            slot_to_elicit='ContentData',
            message={
                'contentType': 'PlainText',
                'content': '🔖예시: 2016년도 출생아 수와 노년 인구의 수를 알려줘.'
            }
        )
    
    response = retrieve(content_data, 'haiku',intent_request, session_attributes)
    text_response = response['output']['text'] if 'output' in response and 'text' in response['output'] else 'No response found.'
    source_text, source_location = extract_citation_data(response)
    content = "{}\n\n(출처: {})".format(text_response, source_location) if source_location else text_response


    return build_response(
        intent_request=intent_request,
        session_attributes=session_attributes,
        fulfillment_state="Fulfilled",
        message={
            'contentType': 'PlainText',
            'content': content
        }
    )

##########################################################################
# TODO Shuan 진행중
# AWS S3에서 문서의 내용을 읽어오는 함수
def get_document_text(bucket, key):
    s3 = boto3.client('s3')

    try:
        # S3에서 PDF 파일 내용을 가져옴
        response = s3.get_object(Bucket=bucket, Key=key)
        byte_content = response['Body'].read()
        
        # BytesIO 객체를 통해 바이트 내용을 파일처럼 처리
        file_stream = BytesIO(byte_content)
        
        # PdfReader를 사용하여 PDF 파일 읽기
        reader = PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            # PDF의 각 페이지에서 텍스트를 추출
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error getting document text from PDF: {e}")
        raise



def handle_doc_summary(intent_request, session_attributes):
    s3_path = None
    file_name = None
    logger.info('session_attributes: %s', session_attributes)

    # 세션 속성에서 'userFilesUploaded'의 값을 파싱하여 s3 경로와 파일 이름을 추출합니다.
    user_files_uploaded_str = session_attributes.get('userFilesUploaded')
    if user_files_uploaded_str:
        try:
            # JSON 문자열 파싱
            user_files_uploaded = json.loads(user_files_uploaded_str)
            if user_files_uploaded:
                # 첫 번째 파일 정보 추출
                first_file_info = user_files_uploaded[0]
                s3_path = first_file_info['s3Path']
                file_name = first_file_info['fileName']
                logger.info('s3_path: %s', s3_path)
                logger.info('file_name: %s', file_name)
        except json.JSONDecodeError as e:
            logger.error('JSON 파싱 에러: %s', e)
            return close(
                intent_request,
                session_attributes,
                'Failed',
                {'contentType': 'PlainText', 'content': '파일 정보 파싱에 실패했습니다.'}
            )

    # 파일 정보가 없으면 사용자에게 문서 업로드를 요청합니다.
    if not s3_path or not file_name:
        return elicit_slot(
            session_attributes=session_attributes,
            intent_name=intent_request['sessionState']['intent']['name'],
            slots=get_slots(intent_request),
            slot_to_elicit='Document',
            message={
                'contentType': 'PlainText',
                'content': '문서를 첨부하고, 원하시는 작업을 입력해주세요.'
            }
        )

        
    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])  # 예: 'us-east-1:186ad67f-ce32-ce73-e80e-369b01fd2c21/문서명.pdf'
    
    logger.info('s3_path: %s, file_name: %s', s3_path, file_name) #  s3://kb-able-talk-s3/us-east-1:186ad67f-ce32-ce73-e80e-369b01fd2c21/저출산-1713403418116.pdf, file_name: 저출산.pdf
    logger.info('Bucket: %s, Key: %s', bucket, key) # Bucket: kb-able-talk-s3, Key: us-east-1:186ad67f-ce32-ce73-e80e-369b01fd2c21/저출산-1713403418116.pdf


    
    # 문서 분석에 필요한 S3 경로 및 키를 추출하고 Bedrock의 retrieve 함수를 호출
    if s3_path and file_name:
        # s3 경로에서 버킷 이름과 키 추출
        bucket = s3_path.split('/')[2]          # 예: 'kb-able-talk-s3'
        key = '/'.join(s3_path.split('/')[3:])  # 예: 'us-east-1:186ad67f-ce32-ce73-e80e-369b01fd2c21/문서명.pdf'
        
        
        # Bedrock의 retrieve_and_generate 메소드를 호출하여 문서 요약 생성
        # 여기서 content_text는 S3에서 추출한 문서의 내용을 나타내며, 이 내용은 실제 코드에서 S3에서 문서를 읽어와야 합니다.
        # 예제에서는 'content_text' 변수가 문서의 내용을 가지고 있다고 가정합니다.
        content_text = get_document_text(bucket, key)  # 이 함수는 S3로부터 문서 내용을 가져와야 합니다. ########################################
        logger.info('content_text: %s', content_text)
        
        
        
        
        
        
        
        
        
        bedrock_response = retrieve(content_text, "haiku", intent_request, session_attributes)
        summary = bedrock_response['output']['text'] if 'output' in bedrock_response and 'text' in bedrock_response['output'] else 'No response found.'
        
        # 요약된 내용을 사용자에게 반환하는 로직을 구현
        # 이 부분은 응답을 사용자에게 보내는 방식에 따라 달라질 수 있습니다.
        return {
            'contentType': 'PlainText',
            'content': summary
        }
    else:
        # S3 경로 또는 파일 이름이 없을 경우 사용자에게 알림
        return elicit_slot(
            session_attributes,
            intent_name=intent_request['sessionState']['intent']['name'],
            slots=get_slots(intent_request),
            slot_to_elicit='Document',
            message={
                'contentType': 'PlainText',
                'content': '(1) 먼저 문서를 첨부해주세요. (2) 원하시는 작업을 입력해주세요.'
            }
        )
        
        

    
    # try:
    #     # S3에서 파일 내용 가져오기
    #     response = s3.get_object(Bucket=bucket, Key=key)
    #     content_data = response['Body'].read()
    #     encoding = chardet.detect(content_data)['encoding']
    #     content_text = content_data.decode(encoding)
    #     logger.info('content_data: %s', content_text)
        
        
    #     # Bedrock의 retrieve_and_generate 메소드를 호출하여 문서 요약 생성
    #     bedrock_response = retrieve(content_text, "QRU0YV9GL5")
    #     summary = bedrock_response['output']['text'] if 'output' in bedrock_response and 'text' in bedrock_response['output'] else 'No response found.'
        
    #     # 문서 요약 내용을 Lex로 반환
    #     return build_response(
    #         intent_request,
    #         session_attributes,
    #         'Fulfilled',
    #         summary
    #     )

    # except Exception as e:
    #     logger.error(f"예외 처리: {e}")
    #     # 예외 처리 및 사용자에게 메시지 전달
    #     return close(
    #         intent_request,
    #         session_attributes,
    #         'Failed',
    #         {'contentType': 'PlainText', 'content': '문서 처리 중 예상치 못한 오류가 발생했습니다.'}
    #     )




        


########################### 요약(S) ###########################
def handle_summary(intent_request, content_data, session_attributes):
    if not content_data:
        return elicit_slot(
            session_attributes=session_attributes,
            intent_name='Reception',
            slots=get_slots(intent_request),
            slot_to_elicit='ContentData',
            message={
                'contentType': 'PlainText',
                'content': '요약할 내용을 입력해주세요:'
            }
        )
    
    prompt = f"다음 내용을 요약해주세요:\n{content_data}"
    
    try:
        response = retrieve(prompt, 'haiku', intent_request, session_attributes)
        text_response = response['output']['text'] if 'output' in response and 'text' in response['output'] else '요약 결과를 생성할 수 없습니다.'
        
        return build_response(
            intent_request=intent_request,
            session_attributes=session_attributes,
            fulfillment_state="Fulfilled",
            message={
                'contentType': 'PlainText',
                'content': text_response
            }
        )
    
    except ClientError as e:
        logger.error(f"Error calling Bedrock retrieve_and_generate: {e}")
        return fallbackIntent(intent_request)


########################### 번역(S) ###########################
def handle_translation(intent_request, content_data, translation_language, session_attributes):
    # 번역하려는 언어가 유효한지 검사합니다.
    valid_languages = ["영어", "한글", "일본어", "English", "Korean", "Japanese"]
    if not translation_language or translation_language not in valid_languages:
        # 유효하지 않은 경우, 사용자에게 TranslationLanguage 슬롯을 다시 요청합니다.
        return elicit_slot(
            session_attributes=session_attributes,
            intent_name='Reception',
            slots=get_slots(intent_request),
            slot_to_elicit='TranslationLanguage',
            message={
                'contentType': 'PlainText',
                'content': '어떤 언어로 번역할까요? (영어, 한글, 일본어)'
            }
        )
    
    # 번역할 내용이 없으면 사용자에게 요청합니다.
    if not content_data:
        return elicit_slot(
            session_attributes=session_attributes,
            intent_name='Reception',
            slots=get_slots(intent_request),
            slot_to_elicit='ContentData',
            message={
                'contentType': 'PlainText',
                'content': '번역할 내용을 입력해주세요:'
            }
        )
    
    translation_direction = get_translation_direction(translation_language)
    prompt = f"다음 내용을 {translation_direction} 번역해주세요:\n{content_data}"
    
    try:
        response = retrieve(prompt, 'haiku', intent_request, session_attributes)
        text_response = response['output']['text'] if 'output' in response and 'text' in response['output'] else '번역 결과를 생성할 수 없습니다.'
        
        return build_response(
            intent_request=intent_request,
            session_attributes=session_attributes,
            fulfillment_state="Fulfilled",
            message={
                'contentType': 'PlainText',
                'content': text_response
            }
        )
    
    except ClientError as e:
        logger.error(f"Error calling Bedrock retrieve_and_generate: {e}")
        return fallbackIntent(intent_request)

def get_translation_direction(translation_language):
    language_lower = translation_language.lower()
    if language_lower in ["korean", "한국어", "한글"]:
        return "한국어로"
    elif language_lower in ["english", "영어"]:
        return "영어로"
    elif language_lower in ["japanese", "일본어", "일본"]:
        return "일본어로"
    else:
        return "영어로"


########################### 문구생성 (S) ###########################
def handle_gen_text(intent_request, content_data, session_attributes):
    if not content_data:
        return elicit_slot(
            session_attributes=session_attributes,
            intent_name='Reception',
            slots=get_slots(intent_request),
            slot_to_elicit='ContentData',
            message={
                'contentType': 'PlainText',
                'content': '요구사항을 알려주세요. 🔖예시: KB증권의 VIP고객을 대상으로 감사 인사 메세지를 작성해줘.'
            }
        )
    
    prompt = f"다음 내용으로 문구를 생성해주세요. 적절한 이모지도 사용해주세요.\n{content_data}"
    
    try:
        response = retrieve(prompt, 'sonnet', intent_request, session_attributes)
        text_response = response['output']['text'] if 'output' in response and 'text' in response['output'] else '문구를 생성할 수 없습니다.'
        
        return build_response(
            intent_request=intent_request,
            session_attributes=session_attributes,
            fulfillment_state="Fulfilled",
            message={
                'contentType': 'PlainText',
                'content': text_response
            }
        )
    
    except ClientError as e:
        logger.error(f"Error calling Bedrock retrieve_and_generate: {e}")
        return fallbackIntent(intent_request)


########################### SQL (S) ###########################
def handle_gen_sql(intent_request, content_data, session_attributes):
    if not content_data:
        return elicit_slot(
            session_attributes=session_attributes,
            intent_name='Reception',
            slots=get_slots(intent_request),
            slot_to_elicit='ContentData',
            message={
                'contentType': 'PlainText',
                'content': '요구사항을 알려주세요. 🔖예시: 식품의 정보를 담은 테이블과 식품의 주문 정보를 담은 테이블이 있습니다. 테이블에서 생산일자가 2024년 1월인 식품들의 총매출을 조회하는 SQL문을 작성해주세요. 이때 결과는 총매출을 기준으로 내림차순 정렬해주시고 총매출이 같다면 식품 ID를 기준으로 오름차순 정렬해주세요.'
            }
        )

    prompt = f"당신은 경력 15년차 Database Administrator입니다. 다음의 내용으로 올바르게 동작하는 SQL을 작성해주세요. 그리고 작성한 SQL에 대해 간략하게 설명해주세요.\n\n{content_data}"

    try:
        response = retrieve(prompt, 'haiku', intent_request, session_attributes)
        text_response = response['output']['text'] if 'output' in response and 'text' in response['output'] else 'SQL을 생성할 수 없습니다.'
        
        return build_response(
            intent_request=intent_request,
            session_attributes=session_attributes,
            fulfillment_state="Fulfilled",
            message={
                'contentType': 'PlainText',
                'content': text_response
            }
        )
    
    except ClientError as e:
        logger.error(f"Error calling Bedrock retrieve_and_generate: {e}")        
        return fallbackIntent(intent_request)


########################### 기타 (S) ###########################
def handle_etc(intent_request, content_data, session_attributes):
    if not content_data:
        return elicit_slot(
            session_attributes=session_attributes,
            intent_name='Reception',
            slots=get_slots(intent_request),
            slot_to_elicit='ContentData',
            message={
                'contentType': 'PlainText',
                'content': '어떤 것을 도와드릴까요? 🔖예시: 저녁에 먹는 사과는 몸에 해로운가요?'
            }
        )
    
    prompt = f"당신은 사람들에게 도움이 되는 조수 입니다. 질문에 대해 확인하고 적절하게 답변해줘.\n\n{content_data}"
                      
    try:
        response = retrieve(prompt, 'haiku', intent_request, session_attributes)
        text_response = response['output']['text'] if 'output' in response and 'text' in response['output'] else 'SQL을 생성할 수 없습니다.'
        
        return build_response(
            intent_request=intent_request,
            session_attributes=session_attributes,
            fulfillment_state="Fulfilled",
            message={
                'contentType': 'PlainText',
                'content': text_response
            }
        )
    
    except ClientError as e:
        logger.error(f"Error calling Bedrock retrieve_and_generate: {e}")        
        return fallbackIntent(intent_request)
    
    

def fallbackIntent(intent_request):
    try:
        session_attributes = get_session_attributes(intent_request)
        user_content = intent_request['inputTranscript']
        response = retrieve(user_content, 'haiku', intent_request, session_attributes)
        text_response = response['output']['text'] if 'output' in response and 'text' in response['output'] else 'No response found.'
    
        return build_response(
            intent_request=intent_request,
            session_attributes=session_attributes,
            fulfillment_state="Fulfilled",
            message={
                'contentType': 'PlainText',
                'content': text_response
            }
        )

    except Exception as e:
        logger.error('Exception: %s', e, exc_info=True)
        
        return build_response(
            intent_request=intent_request,
            session_attributes=session_attributes,
            fulfillment_state="Failed",
            message={
                'contentType': 'PlainText',
                'content': '오류가 발생했습니다. 다시 시도해주세요.'
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
        
        # 대화 상태와 슬롯 값을 가져옴
        session_attributes = get_session_attributes(intent_request)
        task_type = get_slot(intent_request, 'TaskType').lower()
        content = get_slot(intent_request, 'ContentData')
        
        # '번역' 태스크에 대한 슬롯 값 가져오기
        translation_language = get_slot(intent_request, 'TranslationLanguage') if task_type == '번역' else None
        
        # 각 태스크 유형별로 처리할 핸들러 함수를 사전에 매핑
        task_handlers = {
            '요약': handle_summary,
            '번역': handle_translation,
            '문구생성': handle_gen_text,
            'sql': handle_gen_sql,
            '기타': handle_etc,
            '문서리뷰': handle_doc_summary,
            'rag': handle_rag
        }
        
        if task_type == '번역':
            return task_handlers[task_type](intent_request, content, translation_language, session_attributes)
        elif task_type == '문서리뷰':
            return task_handlers[task_type](intent_request, session_attributes)
        else:
            return task_handlers[task_type](intent_request, content, session_attributes)
    
    except KeyError:
        # 필요한 슬롯 값이 없을 경우 fallbackIntent 함수 호출
        logger.warning("KeyError occurred while accessing slot values. Calling fallbackIntent.")
        return fallbackIntent(intent_request)
    
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        return fallbackIntent(intent_request)


def dispatch(intent_request):
    intent_name = intent_request['sessionState']['intent']['name']
    response = None
    
    if intent_name == 'Reception':
        return Reception(intent_request)
    elif intent_name == 'FallbackIntent':
        return fallbackIntent(intent_request)

    raise Exception('Intent with name ' + intent_name + ' not supported')

def lambda_handler(event, context):
    try:
        logger.info('Event: %s', json.dumps(event))
        
        # TODO S3 업로드 이벤트 처리
        # 업로드 이벤트를 받으면, 해당 문서를 OpenSearch에 임베딩하는 로직 필요
        # for record in event['Records']:
        #     bucket_name = record['s3']['bucket']['name']
        #     object_key = record['s3']['object']['key']
            
        #     # todo: S3 버킷을 스캔하고, 버킷 내의 모든 문서에 대한 메타데이터를 수집하여 metadata.json 파일을 생성한다.
        #     # todo: amazon EventBridge를 통해 
            
        #     document_text = get_document_text(bucket_name, object_key)
            
        #     # Bedrock API를 사용하여 문서를 처리합니다.
        #     bedrock_response = call_bedrock_api(document_text)
            
        #     # 처리된 결과를 OpenSearch에 임베딩합니다.
        #     index_document_in_opensearch(bedrock_response)
        
        
        response = dispatch(event)
        return response
    except Exception as e:
        # 모든 예외는 여기에서 처리되며, 사용자에게 친절한 메시지를 반환합니다.
        return handle_exception(e, event, get_session_attributes(event))
    
    
    
    
