import csv

dicts = []
with open('hubble-birthdays-full-year.csv') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    count = 0
    for row in csv_reader:
        dicts.append(row)
        count += 1
    print(dicts)
    print(next(item for item in dicts if item['Date'] == '16-Dec'))

with open('csv_output.csv', mode = 'w', newline = '') as csvfile:
    headers = ['Date', 'Year', 'Name', 'Caption', 'URL']
    csv_writer = csv.DictWriter(csvfile, fieldnames = headers)
    csv_writer.writeheader()
    csv_writer.writerows(dicts)