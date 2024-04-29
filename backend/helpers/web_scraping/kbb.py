import requests
from bs4 import BeautifulSoup
import csv


start_line = 671
end_line = 3500


with open('out.csv', mode='w', newline='') as csvfile:
    fieldnames = ['Make', 'Model', 'Year', 'review']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    with open('Final.csv', newline='') as car_csvfile:
        reader = csv.DictReader(car_csvfile)
        for i, row in enumerate(reader, start=1):
          if i < start_line:
              continue
          if i > end_line:
              break
          
          make = row['Make']
          model = row['Model']
          year = row['Year']

          url = f'https://www.kbb.com/{make}/{model}/{year}/consumer-reviews/'
          response = requests.get(url)

          if response.status_code == 200:
              soup = BeautifulSoup(response.text, 'html.parser')
              paragraphs = soup.find_all('p', class_='css-1os3rsl emgezi80')

              for p in paragraphs:
                  writer.writerow({'Make': make, 'Model': model, 'Year': year, 'review': p.text})
              print(f"Finished getting reviews for {make} {model} {year}")

              
          else:
              print(f"Failed to fetch the webpage for {make} {model} {year}. Status code: {response.status_code}")
