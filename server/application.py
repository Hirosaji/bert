# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys

sys.path.append("./apigw")
from settings import APIGW
from apigw import (
    get_apigw_client,
    request_search_article,
)
from util import is_japanese

sys.path.append("./bert_script")
from params import BERT_PRAMS
from extract_features import get_futures

import numpy as np
import codecs
import tensorflow as tf
import json


def checkPwd():
    with open("/efs/sample_text.txt") as f:
        output = f.read()
    return output


def cos_simularity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def calc_simlarity(total, each):
    sims = [
        cos_simularity(e["layers"][0]["values"], total["layers"][0]["values"])
        for e in each
    ]
    return sims


class convert_to_simlarity:

    def __init__(self):
        self.output = {
            "body": None,
            "title": None,
            "link": None,
            "sim": None
        }

    def from_kijiID(self, kiji_ID):
        # get client
        client = get_apigw_client(APIGW["CLIENT_ID"], APIGW["CLIENT_SECRET"])

        # create request object (GET method)
        r = request_search_article(client, kiji_ID)

        raw_title = r.json()["hits"][0]["title"]
        self.output["title"] = list(filter(is_japanese, raw_title.split()))
        self.output["link"] = "https://www.nikkei.com/article/" + r.json()["hits"][0]["kiji_id_enc"]

        raw_body = r.json()["hits"][0]["body"]
        self.output["body"] = list(filter(is_japanese, raw_body.split()))

        # body_ja = [
        #     "宅配便最大手のヤマトホールディングス（HD）の業績回復が遅れている。2019年4～12月期の連結営業利益は500億円台半ばと、前年同期比3割弱減ったようだ。ネット通販で大口顧客が離反した状況を解消できず、宅配便の取扱数が減少した。主戦場のネット通販で後手に回り、17年10月からの値上げを利益につなげられない構図が固定化しつつある。",
        #     "10～12月期の宅配便の取扱数は3%弱少ない約5億1千万個、4～12月期は0.7%減の約13億9千万個だった。ヤマトは人手不足で配送能力が限界に達したと判断し、17年10月から値上げに合わせて、1年間顧客からの宅配便受注を抑制していた。ただその間に大口顧客がよりコストの安いライバル社への発注を増やし、ヤマトの存在感が以前と比べて低下している。",
        #     "10～12月期は主力の宅配便事業が減収減益となった可能性がある。値上げ効果を宅配便取扱数の減少が相殺し、コスト削減も追いつかなかった。ヤマトは昨年10月末に今期の予想営業利益を前期比6%増の620億円に下方修正した。この業績予想は宅配便の取扱数を2%強増やす前提となっており、閑散期にあたる1～3月での挽回は容易ではないとの見方が市場では多い。",
        #     "同社は長距離を運ぶ幹線トラックの積載率を向上させるなど、コスト削減に向けた管理強化を進めている。新設した夕方以降の配送組織「アンカーキャスト」の配送効率が上昇するなど明るい兆しもある。ただ宅配便の安定増はまだ見込めておらず、苦しい事業環境がまだ続きそうだ。"
        # ]

        # run BERT
        self.output["sims"] = self.kijiBody2similarity()

        return self.output


    def from_kijiBody(self, kiji_body):
        self.output["body"] = kiji_body.splitlines()

        # run BERT
        self.output["sims"] = self.kijiBody2similarity()

        return self.output


    def kijiBody2similarity(self):
        kiji_body = self.output["body"]
        input_data = ["".join(kiji_body)] + kiji_body
        raw_features = get_futures(BERT_PRAMS, input_data)

        cls_features = []
        for raw_feature in raw_features:
            cls_feature = list(filter(lambda layer: layer["token"] == "[CLS]", raw_feature["features"]))
            cls_features.append(cls_feature[0])

        simlarities = calc_simlarity(cls_features[0], cls_features[1:])

        return simlarities


# EB looks for an 'application' callable by default.
application = Flask(__name__)
application.config["JSON_AS_ASCII"] = False

# set convert_to_simlarity class
req = convert_to_simlarity()

# enable CORS
CORS(application)

# check post method behavior
application.add_url_rule(
    "/",
    "index",
    (
        lambda: jsonify(
            {
                "data": [
                    {"item": "total", "ratio": 0.75},
                    {"item": "1st", "ratio": 0.57},
                    {"item": "2nd", "ratio": 0.23},
                    {"item": "3rd", "ratio": 0.56},
                    {"item": "4th", "ratio": 0.95},
                    {"item": "5th", "ratio": 0.81},
                ]
            }
        )
    ),
    methods=["GET"],
)

# check whether there is mount file
application.add_url_rule("/pwd", "pwd", (lambda: jsonify({"pwd": checkPwd()})))

# # test bert sample
# application.add_url_rule(
#     "/sample", "sample", (lambda: jsonify({"body": pred(),})), methods=["GET"],
# )

# evaluate article by kiji_ID
application.add_url_rule(
    "/kiji_ID",
    "kiji_ID",
    (
        lambda: jsonify(
            {
                "id": request.get_json()["kiji_ID"],
                "context": req.from_kijiID(request.get_json()["kiji_ID"]),
            }
        )
    ),
    methods=["POST"],
)

# evaluate article by kiji_body
application.add_url_rule(
    "/kiji_body",
    "kiji_body",
    (
        lambda: jsonify(
            {
                "id": "",
                "context": req.from_kijiBody(request.get_json()["kiji_body"]),
            }
        )
    ),
    methods=["POST"],
)

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()
