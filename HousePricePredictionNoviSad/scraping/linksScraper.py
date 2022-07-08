from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.expected_conditions import presence_of_element_located
import pandas as pd
import csv
import time
import copy
from selenium.webdriver.chrome.options import Options

PATH = '/Users/djord/Downloads/chromedriver1.exe'
OPTIONS = 'C:\Program Files\Google\Chrome Beta\Application\chrome.exe'
ITERATE_FINISH = 201
ITERATE_START = 1

if __name__ == '__main__':
    s = Service(PATH)
    options = Options()
    options.binary_location = OPTIONS
    driver = webdriver.Chrome(options=options, service=s)
    for page in range(ITERATE_START, ITERATE_FINISH):
        url = 'https://www.4zida.rs/prodaja-stanova/novi-sad'
        page_num = '?strana=' + str(page)
        driver.get(url + page_num)
        # try:
        #     main = WebDriverWait(driver,10).until(
        #         EC.presence_of_element_located((By.CLASS_NAME, "app-body"))
        #     )
        #     print(main.text)
        links = driver.find_elements(By.CLASS_NAME, "ad-preview-card")
        hrefs = []
        for link in links:
            pick = link.find_element(By.CLASS_NAME, "meta-container")
            # print(pick.text)
            elem = pick.find_element(By.TAG_NAME, "a")
            href = elem.get_attribute('href')
            print(href)
            hrefs.append(href)
        time.sleep(1)
        with open('../resources/links.csv', 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            for hr in hrefs:
                writer.writerow([hr])
    driver.close()
