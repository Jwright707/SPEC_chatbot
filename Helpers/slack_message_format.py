import os


def unidentified_format(user_question):
    return {
        'channel': os.getenv("DOTTY_CHANNEL_ID"),
        'username': 'Dotty the Chatbot',
        'icon_emoji': '',
        'blocks': [
            {
                'type': 'header',
                'text': {
                    'type': 'plain_text',
                    'text': "Question Assistance "
                }
            },
            {
                'type': 'divider'
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Question: *{user_question}*"
                }
            }
        ]
    }


def response_format(user_answer):
    return {
        'channel': os.getenv("DOTTY_CHANNEL_ID"),
        'username': 'Dotty the Chatbot',
        'icon_emoji': '',
        'blocks': [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Answer: *{user_answer}*"
                }
            },
            {
                'type': 'section',
                'text': {
                    "type": "plain_text",
                    "text": "Thank you for your help, I'm learning more everyday!",
                }
            },
            {
                'type': 'divider'
            },
        ]
    }


def ignore_format():
    return {
        'channel': os.getenv("DOTTY_CHANNEL_ID"),
        'username': 'Dotty the Chatbot',
        'icon_emoji': '',
        'blocks': [
            {
                'type': 'section',
                'text': {
                    "type": "plain_text",
                    "text": "Thank you, question has successfully been ignore!",
                }
            },
            {
                'type': 'divider'
            },
        ]
    }


def no_question():
    return {
        'channel': os.getenv("DOTTY_CHANNEL_ID"),
        'username': 'Dotty the Chatbot',
        'icon_emoji': '',
        'blocks': [
            {
                'type': 'section',
                'text': {
                    "type": "plain_text",
                    "text": "There are currently no questions to ignore!",
                }
            },
            {
                'type': 'divider'
            },
        ]
    }
