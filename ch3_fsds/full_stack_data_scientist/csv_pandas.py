import pandas

pandas_df = pandas.read_csv('hubble-birthdays-full-year.csv')
print(pandas_df)

pandas_df = pandas.read_csv('hubble-birthdays-full-year.csv', index_col = 'Date')
print(pandas_df)

pandas_df = pandas.read_csv('hubble-birthdays-full-year.csv', index_col = 'Date', dtype = {'Year':'Int32'})
print(pandas_df)

pandas_df = pandas.read_csv('hubble-birthdays-full-year.csv', index_col = 'Birthday', header = 0,  names = ['Birthday', 'Year of observation', 'Name of image', 'Caption', 'Image URL'])
print(pandas_df)
pandas_df.to_csv('pandas_output.csv')