import os
import time
import requests
import json


def HttpRequest(apiname, controlname, operate, jtype, inputdata):
    headers = {"content-Type": "application/json"}
    datas = json.JSONEncoder().encode(inputdata)
    url = "http://192.168.43.214:88/api/" + apiname + "/" + controlname + "/" + operate
    if jtype == "get":
        r = requests.get(url, data=datas, headers=headers)
    if jtype == "post":
        r = requests.post(url, data=datas, headers=headers)

    jsonobj = json.loads(r.text)
    return jsonobj


def upload_img(filename):
    jsonobj = HttpRequest("thirdparty", "oss", "policy", "get", {})
    host = jsonobj['data']["host"]
    dir = jsonobj['data']["dir"]
    filebasename = os.path.basename(filename)
    policy = jsonobj['data']["policy"]
    signature = jsonobj['data']["signature"]
    OSSAccessKeyId = jsonobj['data']["accessid"]

    file = {"file": open(filename, "rb")}

    r = requests.post(host,
                      data={
                          "key": dir + str(time.time()) + "_" + filebasename,
                          "dir": dir,
                          "policy": policy,
                          "signature": signature,
                          "OSSAccessKeyId": OSSAccessKeyId,
                          "host": host
                      },
                      files=file)


# dict = {}
# dict["param"] = "yyl"
# print(getjson("yuangong", "yuangonginfo", "test", dict, "post")["msg"])

upload_img("D:\\whg\\laorenyouhao\\face_collection\\images\\105\\blink_1.jpg")
