import requests
from bs4 import BeautifulSoup
import csv

def scrape_kbb_ratings(make, model, year):
  url = f"https://www.kbb.com/{make}/{model}/{year}/consumer-reviews/"
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')
  rating_divs = soup.find_all('div', class_='css-wti69m')
  ratings = [rating_div.text for rating_div in rating_divs]
  return ratings

with open('Final.csv', mode='r') as file:
    reader = csv.DictReader(file)
    cars = list(reader)

with open('something.csv', mode='w', newline='') as file:
    fieldnames = ["Make", "Model", "Year", "Ratings"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    start = 1
    for car in cars:
        make = car["Make"]
        model = car["Model"]
        year = car["Year"]
        ratings = scrape_kbb_ratings(make, model, year)
        writer.writerow({"Make": make, "Model": model, "Year": year, "Ratings": ratings})
        print(start)
        start+=1
