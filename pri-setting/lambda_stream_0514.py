import boto3
import os
import json
from asyncio import get_event_loop, gather, sleep
import botocore.exceptions
import openai
import random
import decimal
import logging
import chardet
import PyPDF2
from PyPDF2 import PdfReader
from io import BytesIO
from urllib.parse import unquote
from botocore.exceptions import ClientError
import gzip
import csv

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AWS_REGION = os.environ["AWS_REGION"]
ENDPOINT_URL = os.environ.get("ENDPOINT_URL", f'https://bedrock-runtime.{AWS_REGION}.amazonaws.com')
modelId = "anthropic.claude-3-haiku-20240307-v1:0"

accept = "application/json"
contentType = "application/json"

dynamodb_client = boto3.resource('dynamodb')
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=AWS_REGION, endpoint_url=ENDPOINT_URL)
s3 = boto3.client('s3')

def lambda_handler(event, context):
    print('event: ', event)
    response = router(event, context)
    return response

def router(event, context):
    intent_name = event['sessionState']['intent']['name']
    sess_id = event['sessionId']
    result = get_event_loop().run_until_complete(openai_async_api_handler(event, context))
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
