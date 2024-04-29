import csv
import requests
from bs4 import BeautifulSoup


### THIS IS FOR THE BATCHES, 
# Define the start and end index for the rows to process
start_index = 3178
end_index = 3500

with open('Final.csv', mode='r') as file:
    reader = csv.DictReader(file)
    
    with open('out.csv', mode='w', newline='') as output_file:
        fieldnames = ['Make', 'Model', 'Year', 'Standardized colors', 'Image links']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in enumerate(reader, start=1):
            if index < start_index:
                continue  
            if index >= end_index:
                break  

            make = row['Make']
            model = row['Model']
            year = row['Year']
            colors = row['Standardized colors'].split(', ')

            image_links = []

            for color in colors:
                formatted_query = '+'.join((make + ' ' + model + ' ' + year + ' ' + color).split())
                url = f"https://www.google.com/search?hl=en&tbm=isch&q={formatted_query}"
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                images = soup.find_all("img")

                image_count = 0

                # second image is the first image technically (counts google logo as #1)
                for image in images:
                    image_source = image.get("src")
                    if image_source:
                        image_count += 1
                        if image_count == 2:
                            image_links.append(image_source)
                            break

            writer.writerow({
                'Make': make,
                'Model': model,
                'Year': year,
                'Standardized colors': row['Standardized colors'],
                'Image links': image_links
            })

            print(f"Finished processing car {index}: {make} {model} {year} - {len(image_links)} color images found")

print("Finished writing to CSV file.")
