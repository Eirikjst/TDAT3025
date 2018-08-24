from contextlib import closing
import codecs
import csv
import numpy as np
import requests as requests

# For importing and using ReadCSV in other project
#
# path = ''
# url = ''
#
# import importlib.machinery
# loader = importlib.machinery.SourceFileLoader('Read_csv_files', path)
# csv = loader.load_module('Read_csv_files')
#
# data = csv.ReadCSV(url).read_csv_from_url()

class ReadCSV:
    def __init__(self, url_or_filepath):
        self.url_or_filepath = url_or_filepath

    def read_csv_from_url(self):
        with closing(requests.get(self.url_or_filepath, stream=True)) as r:
            reader = csv.reader(codecs.iterdecode(r.iter_lines(), 'utf-8'), delimiter=',', quotechar='"')
            next(reader, None)
            data = [data for data in reader]
        result = np.asarray(data, dtype=str).astype(np.float32)
        return result
    
    #To be continued
    def read_csv_from_file(self):
        return None
