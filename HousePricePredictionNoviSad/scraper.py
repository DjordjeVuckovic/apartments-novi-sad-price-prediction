import copy

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import csv
import time
import json
import pickle
import pandas as pd
from selenium.webdriver.chrome.options import Options

PATH = '/Users/djord/Downloads/chromedriver1.exe'
OPTIONS = 'C:\Program Files\Google\Chrome Beta\Application\chrome.exe'

if __name__ == '__main__':
    options = Options()
    options.binary_location = OPTIONS
    s = Service(PATH)
    driver = webdriver.Chrome(options=options, service=s)
    dictionary = {}
    table_data = []
    with open('resources/links.csv', 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row:
                print(row)
                try:
                    driver.get(row[0])
                    elems = driver.find_elements(By.CLASS_NAME, "value")
                    labels = driver.find_elements(By.CLASS_NAME, "label")
                    address = driver.find_element(By.CLASS_NAME, "address")
                    counter = 0
                    # time.sleep(1)
                    for i in range(len(elems)):
                        dictionary['Adresa:'] = address.text
                        dictionary[labels[i].text] = elems[i].text
                        new_dic = copy.deepcopy(dictionary)
                    print(table_data)
                    table_data.append(new_dic)
                except:
                    continue
    print(table_data)

    with open('resources/apart.json', 'w', encoding='utf-8') as f:
        json.dump(table_data, f, sort_keys=True)
    # data_tuples = list(zip(prices[:], prices_mean[:],types[:],areas[:],rooms[:]))  # list of each players name and salary paired together
    # temp_df = pd.DataFrame(data_tuples, columns=['Price', 'PriceMean','Types','Area','Rooms'])  # creates dataframe of each tuple in list
    # temp_df.to_csv('house_prediction.csv')
    # print(temp_df)
    driver.close()
