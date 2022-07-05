# import http.client, urllib.request, urllib.parse, urllib.error, base64
#%%
headers = {
    # Request headers
    'Content-Type': "application/json",
    'Ocp-Apim-Subscription-Key': "843db42db04042f4b82d5f780e54925e",
}
# body = {
#   "i": "HS_M_0010",
#   "r": "all",
#   "p": "default",
#   "ps": "2001-2023",
#   "pc": "default",
#   "spc": True,
#   "fmt": "csv",
#   "mode": "codes",
#   "dec": "default",
#   "off": 0,
#   "max": 10,
#   "head": "M",
#   "lang": 1,
#   "meta": False
# }
# params = urllib.parse.urlencode({})

# # try:
# conn = http.client.HTTPSConnection('api.wto.org')
# s = f"/timeseries/v1/data?{params}"
# print(s)
# conn.request("POST", s, "{body}", headers)
# response = conn.getresponse()
# data = response.read()
# print(eval(data))
# conn.close()
# # except Exception as e:
# #     print(e.errno)
# #     print(e.strerror)
#     # pass

# importing the requests library
import requests, zipfile, io
import pandas as pd

# defining the api-endpoint
API_ENDPOINT = "https://api.wto.org/timeseries/v1/data"

# your API key here
API_KEY = "843db42db04042f4b82d5f780e54925e"

# your source code here
# source_code = '''
# print("Hello, world!")
# a = 1
# b = 2
# print(a + b)
# '''

# data to be sent to api
body = {
		# 'api_option':'paste',
		# 'api_paste_code':source_code,
		# 'api_paste_format':'python'
          "i": "HS_M_0010",
          "r": "all",
          "p": "default",
          "ps": "2001-2023",
          "pc": "default",
          "spc": True,
          "fmt": "csv",
          "mode": "codes",
          "dec": "default",
          "off": 0,
          "max": 500,
          "head": "M",
          "lang": 1,
          "meta": False
        }
#%%
# sending post request and saving response as response object
r = requests.post(url = API_ENDPOINT, json = body, headers=headers)

assert r.status_code == 200, f"{r.status_code}: {r.reason}"
# extracting response text
pastebin_url = r.text
# with open("./data.csv","wb") as f:
#     f.write(r.content)
# data = r.content
print(r.status_code)
print(r.url)
# print(r.content)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(path=".")
#%%
# import requests
url = 'https://comtrade.un.org/api/get?max=5000&type=C&freq=A&px=HS&ps=2018&r=152&p=all&rg=all&cc=851712'
un_data = requests.get(url)
print(un_data.status_code)
pd.DataFrame(eval(un_data.content.decode("utf-8").replace("null","''"))["dataset"])
# %%
