import boto3
import os
import json
from asyncio import get_event_loop, gather, sleep
import botocore.exceptions  # Import botocore.exceptions
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

AWS_REGION = os.environ["AWS_REGION"]
ENDPOINT_URL = os.environ.get("ENDPOINT_URL", f'https://bedrock-runtime.{AWS_REGION}.amazonaws.com')
modelId = "anthropic.claude-instant-v1" 
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
    sessionId = event['sessionId']
    inputTranscript = event['inputTranscript']
    body = json.dumps({"prompt": "Human: " + inputTranscript + "Assistant:", "max_tokens_to_sample": 500})
    session_attributes = get_session_attributes(event)
    print('sessionId ', sessionId)
    print('inputTranscript ', inputTranscript)
    fullreply = ''
    
    if 'streamingDynamoDbTable' in session_attributes and 'streamingEndpoint' in session_attributes:
        apigatewaymanagementapi = boto3.client(
            'apigatewaymanagementapi', 
            endpoint_url = session_attributes['streamingEndpoint']
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
                        text = chunk_obj['completion']
                        fullreply += text
                        print(text)
                        apigatewaymanagementapi.post_to_connection(
                            Data=text.encode('utf-8'),
                            ConnectionId=connectionId
                        )
                    except botocore.exceptions.ClientError as e:  # Correct exception handling
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
        fullreply = response_body["completion"]
    
    message = {
        'contentType': 'CustomPayload',
        'content': fullreply
    }
    fulfillment_state = "Fulfilled"
    return close(event, session_attributes, fulfillment_state, message)

def get_session_attributes(intent_request):
    session_attributes = intent_request['sessionState'].get('sessionAttributes', {})
    # session_attributes['streamingDynamoDbTable'] = 'kb-able-talk-CodeBuildDeploy-1D4T9T228VY1U-streaming'
    # session_attributes['streamingEndpoint'] = 'https://rs75le42c7.execute-api.us-east-1.amazonaws.com/dev/'
    
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
