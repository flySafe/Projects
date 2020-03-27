
const axios = require('axios').default;
function FlightGetRequest(CountryDeparture,CountryArrival,DayDep,MonthDep,YearDep,DayBack,MonthBack,YearBack) {
    url = "https://kiwicom-prod.apigee.net/v2/search?fly_from="+CountryDeparture+"&fly_to="+CountryArrival+"&date_from="+DayDep+"%2F"+MonthDep+"%2F"+YearDep+"&date_to="+DayBack+"%2F"+MonthBack+"%2F"+YearBack
    try {
        var config = {
            headers: { 'apikey': 'm42o4XvdEakkDwDvAsA8KKdheODJQQNX' }
        };

        let content = new Promise((resolve) => {
                resolve(axios.get(url,config))
            })
        return content
    } catch (error) {
        console.error(error);
    }
}

async function GetPrice (CountryDeparture,CountryArrival,DayDep,MonthDep,YearDep,DayBack,MonthBack,YearBack) {
    const buffer = await FlightGetRequest(CountryDeparture,CountryArrival,DayDep,MonthDep,YearDep,DayBack,MonthBack,YearBack)
    let FlightPrice = buffer.data.data[0]['price']
    console.log(FlightPrice)
}

var x = GetPrice('IL','DE','03','05','2020','05','05','2020')
console.log(x)