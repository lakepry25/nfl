"""import requests
import csv
from bs4 import BeautifulSoup

# Get the HTML of the NFL.com WR season stats page
url = "https://www.nfl.com/stats/player-stats/category/receiving/2023/reg/all/receivingreceptions/desc"
response = requests.get(url)

# Create a BeautifulSoup object from the HTML response
soup = BeautifulSoup(response.content, "html.parser")

# Find the table of WR season stats
stats_table = soup.find("table", class_="data-table")

# Extract the stats from the table and save them to a CSV file
with open("nfl_wr_season_stats.csv", "w") as f:
    writer = csv.writer(f)

    # Write the header row
    writer.writerow(["Player", "Team", "Rec", "Yds", "TD"])

    # Iterate over the rows in the table and write them to the CSV file
    for row in stats_table.find_all("tr"):
        player = row.find("td", class_="name").text
        team = row.find("td", class_="team").text
        rec = row.find("td", class_="rec").text
        yards = row.find("td", class_="yds").text
        td = row.find("td", class_="td").text

        writer.writerow([player, team, rec, yards, td])"""

import requests
from bs4 import BeautifulSoup
import csv

# URL of the Pro Football Reference page for receiving stats
url = "https://www.pro-football-reference.com/years/2022/passing.htm"

# Function to scrape data from a given URL
def scrape_page(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'passing'})
        
        # Add print statements to debug
        if table is None:
            print("Table not found on the page.")
            return None, None

        headers = [header.text.strip() for header in table.find_all('th', {'scope': 'col'})]
        data = []
        for row in table.find_all('tr')[1:]:
            row_data = [td.text.strip() for td in row.find_all(['td', 'th'])]
            data.append(row_data)

        return headers, data
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None, None

# Function to write data to a CSV file
def write_to_csv(headers, data, filename):
    with open(filename, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write headers only if the file is empty
        if csv_file.tell() == 0:
            writer.writerow(headers)
        writer.writerows(data)

# Scraping the page
headers, data = scrape_page(url)

if headers is not None and data is not None:
    write_to_csv(headers, data, 'passing_stats_2022.csv')
    print("Scraping and CSV creation completed successfully.")
else:
    print("Error: Unable to scrape data from the page.")
