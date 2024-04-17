import openai
import json
import random
import decimal 
import os
import json
import random
import decimal 
import logging
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

openai.api_key = os.environ['OPENAI_API']

# Amazon S3 클라이언트 생성
s3 = boto3.client('s3')

# Shuan
bedrock_agent_runtime = boto3.client(
    service_name = "bedrock-agent-runtime"
)

# 사실 기반 태스크에는 Haiku를, 창의적 생성 태스크에는 Sonnet을 사용
def retrieve(query, kbId):
    # modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2:1'
    #modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0'
    modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
    
    return bedrock_agent_runtime.retrieve_and_generate(
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

def extract_citation_data(response, llm):
    source_text = ""
    source_location = ""

    if llm == 'claude':
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

    # 다른 LLM의 경우 다른 처리를 할 수 있도록 분기를 추가합니다.
    # 예를 들어:
    elif llm == 'other_llm':
        # 다른 LLM에 대한 처리 로직
        pass

    return source_text, source_location


## TODO Shuan
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
    
    response = retrieve(content_data, "CVGYCPENVU")
    text_response = response['output']['text'] if 'output' in response and 'text' in response['output'] else 'No response found.'
    source_text, source_location = extract_citation_data(response, 'claude')

    # logger.info('response: %s', response)
    logger.info('text_response: %s', text_response)
    logger.info('source_text: %s', source_text)
    logger.info('source_location: %s', source_location)
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
    
    
    
#######################################################


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



def handle_doc_summary(intent_request, session_attributes):
    # 서비스 준비중인 상태 메시지
    service_unavailable_message = {
        'contentType': 'PlainText',
        'content': '서비스 준비중입니다.'
    }
    # 'Close' dialog action을 사용하여 메시지 전달
    return {
        'dialogAction': {
            'type': 'Close',
            'fulfillmentState': 'Fulfilled',
            'message': service_unavailable_message
        },
        'sessionAttributes': session_attributes
    }    
    
###############################################################
# TODO    
def handle_doc_summary2(intent_request, session_attributes):
    # 세션 속성에서 s3 경로와 파일 이름을 추출
    s3_path = session_attributes.get('s3Path')
    file_name = session_attributes.get('fileName')

    # 파일 정보가 없으면 사용자에게 문서 업로드 요청
    if not s3_path or not file_name:
        return elicit_slot(
            session_attributes=session_attributes,
            intent_name=intent_request['sessionState']['intent']['name'],
            slots=get_slots(intent_request),
            slot_to_elicit='Document',
            message={
                'contentType': 'PlainText',
                'content': '문서를 업로드해 주세요.'
            }
        )

    # s3 경로에서 버킷 이름과 키 추출
    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])
    
    try:
        # S3에서 파일 내용 가져오기
        response = s3.get_object(Bucket=bucket, Key=key)
    except s3.exceptions.NoSuchKey:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
        # NoSuchKey 오류에 대한 사용자 친화적인 메시지로 응답
        return close(
            intent_request,
            session_attributes,
            'Failed',
            {'contentType': 'PlainText', 'content': '요청하신 문서를 찾을 수 없습니다. 파일명을 확인하고 다시 시도해 주세요.'}
        )
    except s3.exceptions.ClientError as e:
        logger.error(f"S3 클라이언트 에러: {e}")
        if e.response['Error']['Code'] == 'AccessDenied':
            # 접근 거부 오류에 대한 처리
            return close(
                intent_request,
                session_attributes,
                'Failed',
                {'contentType': 'PlainText', 'content': '문서에 접근할 권한이 없습니다. 권한을 확인하고 다시 시도해 주세요.'}
            )
        else:
            # 기타 S3 클라이언트 오류에 대한 처리
            return close(
                intent_request,
                session_attributes,
                'Failed',
                {'contentType': 'PlainText', 'content': '문서를 불러오는 동안 문제가 발생했습니다. 관리자에게 문의해 주세요.'}
            )
    except Exception as e:
        logger.error(f"예외 처리: {e}")
        # 예상치 못한 예외 처리
        return close(
            intent_request,
            session_attributes,
            'Failed',
            {'contentType': 'PlainText', 'content': '문서 처리 중 예상치 못한 오류가 발생했습니다.'}
        )

    # 문서 내용 읽기 성공
    content_data = response['Body'].read().decode('utf-8')

    try:
        # OpenAI GPT-3을 사용하여 문서 내용 요약
        summary_response = openai.Completion.create(
            model="text-davinci-003",
            prompt=content_data  # 요약 지시문을 프롬프트에 추가하여 더 나은 요약을 얻을 수 있음
        )

        # 요약된 내용 추출
        summary = summary_response.choices[0].text.strip()

        # Lex로 요약된 내용 반환
        return build_response(
            intent_request,
            session_attributes,
            'Fulfilled',
            {'contentType': 'PlainText', 'content': summary}
        )
    except s3.exceptions.NoSuchKey:
        logger.error("지정된 키가 존재하지 않습니다.")
        return close(
            intent_request,
            session_attributes,
            'Failed',
            {'contentType': 'PlainText', 'content': '문서를 찾을 수 없습니다. 문서를 확인하고 다시 시도해 주세요.'}
        )
    except Exception as e:
        # 다른 예외들 처리
        logger.error(f"예외 발생: {e}")
        return close(
            intent_request,
            session_attributes,
            'Failed',
            {'contentType': 'PlainText', 'content': '문서를 처리하는 중 오류가 발생했습니다.'}
        )

###############################################################

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
    
    # 요약할 내용이 있는 경우, 요약 수행
    prompt = [{"role": "system", "content": "다음 내용을 요약해주세요."}, 
                      {"role": "user", "content": content_data}]
    llm_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt
    )

    # 요약된 내용 추출
    response = llm_response.choices[0].message.content

    # 요약된 내용으로 응답 메시지 구성 후 반환
    return build_response(
        intent_request=intent_request,
        session_attributes=session_attributes,
        fulfillment_state="Fulfilled",
        message={
            'contentType': 'PlainText',
            'content': response
        }
    )

def get_translation_direction(translation_language):
    language_lower = translation_language.lower()
    if language_lower in ["korean", "한국어", "한글"]:
        return "to Korean"
    elif language_lower in ["english", "영어"]:
        return "to English"
    elif language_lower in ["japanese", "일본어", "일본"]:
        return "to Japanese"
    else:
        return "to English"
    
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
    prompt = [
        {"role": "system", "content": f"Please translate the following content {translation_direction}."},
        {"role": "user", "content": content_data}
    ]
    
    llm_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt
    )

    response = llm_response.choices[0].message.content

    return build_response(
        intent_request=intent_request,
        session_attributes=session_attributes,
        fulfillment_state="Fulfilled",
        message={
            'contentType': 'PlainText',
            'content': response
        }
    )

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
    
    prompt = [{"role": "system", "content": "다음 내용으로 문구를 생성해주세요. 적절한 이모지도 사용해주세요."}, 
                      {"role": "user", "content": content_data}]
    llm_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt
    )

    response = llm_response.choices[0].message.content

    return build_response(
        intent_request=intent_request,
        session_attributes=session_attributes,
        fulfillment_state="Fulfilled",
        message={
            'contentType': 'PlainText',
            'content': response
        }
    )

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
    
    prompt = [{"role": "system", "content": "당신은 경력 15년차 Database Administrator입니다. 다음의 내용으로 올바르게 동작하는 SQL을 작성해주세요. 그리고 작성한 SQL에 대해 간략하게 설명해주세요."}, 
                      {"role": "user", "content": content_data}]
    llm_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt
    )

    response = llm_response.choices[0].message.content

    return build_response(
        intent_request=intent_request,
        session_attributes=session_attributes,
        fulfillment_state="Fulfilled",
        message={
            'contentType': 'PlainText',
            'content': response
        }
    )

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
    
    prompt = [{"role": "system", "content": "너는 사람들에게 도움이 되는 조수야. 질문에 대해 확인하고 정확하게 답변해줘."}, 
                      {"role": "user", "content": content_data}]
    llm_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt
    )

    response = llm_response.choices[0].message.content

    return build_response(
        intent_request=intent_request,
        session_attributes=session_attributes,
        fulfillment_state="Fulfilled",
        message={
            'contentType': 'PlainText',
            'content': response
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
            '문서리뷰': handle_doc_summary,
            '번역': handle_translation,
            '문구생성': handle_gen_text,
            'sql': handle_gen_sql,
            '기타': handle_etc,
            'rag': handle_rag
        }
            
        if task_type == '번역':
            return task_handlers[task_type](intent_request, content, translation_language, session_attributes)
        elif task_type == '문서리뷰':
            return task_handlers[task_type](intent_request, session_attributes)
        else:
            return task_handlers[task_type](intent_request, content, session_attributes)    
            

        # 처리할 수 없는 태스크 유형이 입력된 경우
        # raise ValueError(f"Unsupported task type: {task_type}")
        
    except Exception as e:
        logger.error('Exception: %s', e, exc_info=True)






def fallbackIntent(intent_request):
    try:
        session_attributes = get_session_attributes(intent_request)
        user_content = intent_request['inputTranscript']
        messages_prompt = [{"role": "system", "content": '너는 사람들에게 도움이 되는 조수야. 질문에 대해 확인하고 친절하고 이해하기 쉽게 답변해줘.'}]
        messages_prompt.append({"role": "user", "content": user_content})
        
        logger.info('messages_prompt: %s', messages_prompt)
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages_prompt)
        logger.info('fallbackIntent response: %s', response)
        
        text = response["choices"][0]["message"]["content"]
        message =  {
                'contentType': 'PlainText',
                'content': text
            }
        fulfillment_state = "Fulfilled"    
        return close(intent_request, session_attributes, fulfillment_state, message)
    except Exception as e:
        logger.error('Exception: %s', e, exc_info=True)


def dispatch(intent_request):
    intent_name = intent_request['sessionState']['intent']['name']
    response = None
    
    if intent_name == 'Reception':
        return Reception(intent_request)
    elif intent_name == 'FallbackIntent':
        return fallbackIntent(intent_request)

    raise Exception('Intent with name ' + intent_name + ' not supported')

def lambda_handler(event, context):
    logger.info('Event: %s', json.dumps(event))
    # logger.info('Function name: %s', context.function_name) #KBSEC
    
    response = dispatch(event)
    return response

# def lambda_handler(event, context):
#     # TODO: 프로그램 로직을 여기에 작성하세요.
    
#     return {
#         'statusCode': 200,
#         'body': json.dumps('Hello from Lambda!')
#     }

