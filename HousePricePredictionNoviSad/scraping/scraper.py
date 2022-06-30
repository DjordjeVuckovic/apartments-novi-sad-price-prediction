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
from selenium.common.exceptions import NoSuchElementException
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
    with open('../resources/finalLinks.csv', 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            print(row)
            try:
                driver.get(row[0])
                elems = driver.find_elements(By.CLASS_NAME, "value")
                labels = driver.find_elements(By.CLASS_NAME, "label")
                address = driver.find_element(By.CLASS_NAME, "address")
                location = driver.find_element(By.CLASS_NAME, "location")
                # lc = driver.find_element(By.CLASS_NAME, "block")
                # lc1 = lc.find_element(By.CLASS_NAME, "flex")
                # lc2 = lc1.find_element(By.TAG_NAME, "span")
                #print(lc1.text)
                time.sleep(1)
                for i in range(len(elems)):
                    dictionary['Lokacija'] = location.text
                    dictionary['Adresa:'] = address.text
                    dictionary[labels[i].text] = elems[i].text
                    new_dic = copy.deepcopy(dictionary)
                print(table_data)
                table_data.append(new_dic)
            # with open('../resources/final3.json', 'w', encoding='utf-8') as f:
            #     json.dump(table_data, f, sort_keys=True)
            except NoSuchElementException:
                time.sleep(2)
    with open('../resources/finalApartments.json', 'w', encoding='utf-8') as f:
        json.dump(table_data, f, sort_keys=True)
    driver.close()
