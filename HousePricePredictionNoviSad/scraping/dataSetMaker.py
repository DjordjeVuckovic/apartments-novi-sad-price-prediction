import pandas as pd
import json


def calculate_nums(price, lists):
    lists.append(price.split()[0])


def calculate_price(price, lists):
    splits = price.split()
    spl = splits[0].split(".")
    # print(spl)
    if len(spl) > 1:
        new_str = spl[0] + spl[1]
    else:
        new_str = spl[0]
    lists.append(new_str)


def calculate_area(elem, areas):
    my = ''
    for s in elem:
        if s == 'm':
            break
        my += s
    areas.append(my)


def calculate_year(elem, years):
    if elem is not None:
        years.append(elem.split("-")[0])
    else:
        years.append('0')


if __name__ == '__main__':
    list_dictionary = {}
    with open("../resources/apart.json", "r", encoding='utf-8') as infile:
        list_dictionary = json.load(infile)
    prices = []
    addresses = []
    rooms = []
    average_price = []
    grs = []
    nams = []
    lifts = []
    stores = []
    areas = []
    evidents = []
    states = []
    availables = []
    yearOfBuild = []
    floors = []
    print(list_dictionary[:10])
    print("Length of map:" + str(len(list_dictionary)))
    for i in range(len(list_dictionary)):
        calculate_price(list_dictionary[i]['Cena:'], prices)
        calculate_nums(list_dictionary[i]['Broj soba:'], rooms)
        addresses.append(list_dictionary[i]['Adresa:'])
        grs.append(list_dictionary[i]['Grejanje:'])
        nams.append(list_dictionary[i]['Nameštenost:'])
        lifts.append(list_dictionary[i]['Lift:'])
        stores.append(list_dictionary[i]['Opremljenost:'])
        evidents.append(list_dictionary[i]['Uknjiženost:'])
        states.append(list_dictionary[i]['Stanje:'])
        availables.append(list_dictionary[i]['Useljivo:'])
        calculate_nums(list_dictionary[i]['Spratnost:'], floors)
        calculate_area(list_dictionary[i]['Površina:'], areas)
        calculate_year(list_dictionary[i].get('Godina izgradnje:'), yearOfBuild)

    # print("Length of prices:" + str(len(prices)))
    # print("Length of addresses:" + str(len(addresses)))
    # print("Length of rooms:" + str(len(rooms)))
    # print("Length of gr:" + str(len(grs)))
    # print("Length of nams:" + str(len(nams)))
    # print("Length of lifts:" + str(len(lifts)))
    # print("Length of stores:" + str(len(stores)))
    # print("Length of areas:" + str(len(areas)))
    print(prices)
    print(addresses)
    print(rooms)
    print(grs)
    print(nams)
    print(lifts)
    print(stores)
    print(areas)
    print(evidents)
    print(states)
    print(availables)
    print(yearOfBuild)
    print(floors)
    data_tuples = list(zip(prices[:], addresses[:], rooms[:], grs[:],
                           nams[:], lifts[:], stores[:], areas[:], evidents[:], states[:], availables[:],
                           yearOfBuild[:], floors[:]))  # list of each players name and salary paired together
    temp_df = pd.DataFrame(data_tuples, columns=['Price(EUR)', 'Address', 'Rooms', 'Heating',
                                                 'SetUp', 'Elevator', 'Stores', 'Area(m2)', 'Evident', 'State',
                                                 'Infrared', 'YearOfBuild',
                                                 'Floor'])  # creates dataframe of each tuple in list
    temp_df.to_csv('dataSet/house_prediction.csv', encoding='utf-8')
    print(temp_df)
