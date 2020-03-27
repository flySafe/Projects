import requests
import json
import sys

'''
Args:
    [0] - CountryDeparture
    [1] - CountryArrival
    [2] - DayDep
    [3] - MonthDep
    [4] - YearDep
    [5] - DayBack
    [6] - MonthDep
    [7] - YearDep
'''

Arguments = sys.argv[1:]
url = "https://kiwicom-prod.apigee.net/v2/search?fly_from="+Arguments[0]+"&fly_to="+Arguments[1]+"&date_from="+Arguments[2]+"%2F"+Arguments[3]+"%2F"+Arguments[4]+"&date_to="+Arguments[5]+"%2F"+Arguments[6]+"%2F"+Arguments[7]
headers = {"apikey": "m42o4XvdEakkDwDvAsA8KKdheODJQQNX"}
request = requests.get(url,headers=headers)
json_data = json.loads(request.text)
with open('Flight.json', 'w') as outfile:
    json.dump(json_data['data'][0], outfile)
