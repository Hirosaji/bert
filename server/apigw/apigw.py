import requests_oauthlib

from oauthlib.oauth2 import BackendApplicationClient

from settings import (
    APIGW,
)

def get_apigw_client(client_id, client_secret):
    """get token and client

    Args:
        client_id (str): apigateway clinent id
        client_secret (str): apigateway clinent secret

    Returns:
       (requests_oauthlib.OAuth2Session):
        https://github.com/requests/requests-oauthlib/blob/master/requests_oauthlib/oauth2_session.py
    """
    client = requests_oauthlib.OAuth2Session(
        client=BackendApplicationClient(client_id)
    )
    client.fetch_token(
        APIGW['TOKEN_ENDPOINT'],
        auth=(client_id, client_secret)
    )
    return client


def request_search_article(client, _kiji_id):
    """search articles with _kiji_id in target article.

    Args:
        client (requests_oauthlib.Oauth2Session): APIGW client
        kiji_id ([str]): kiji_id in target article
    Returns:
        (requests.Response)
    """

    params = {
        'kiji_id': _kiji_id,
        'fields': 'body,kiji_id_enc',
    }
    return client.get(APIGW['SEARCH_V1'], params=params)
