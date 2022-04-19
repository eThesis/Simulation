import numpy as np

# Data class has a single input
# [in] a regular Python array, such as a = [[1,0],[0,1]] then call by A = Data(a)
# Data class converts regular -> numpy array
# Data class passed quality test, no bugs known currently 

class Data:
    def __init__(self, raw_Data_Matrix) :
        self.data_Matrix = np.array(raw_Data_Matrix)
        self.sample_Size, self.dim = self.data_Matrix.shape # each row is one data point

    def getRow(self, row_Number):
        """First row has index = 0"""
        try:
            return self.data_Matrix[row_Number, :]
        except (IndexError, TypeError):
            print('\n⛔️⛔️⛔️ Oops, invalid row number! Request a row between 0 and', str(self.sample_Size - 1), 'inclusive. ⛔️⛔️⛔️\n')

    def getColumn(self, column_Number):
        """First column has index 0"""
        try:
            return self.data_Matrix[:, column_Number]
        except (IndexError, TypeError):
            print('\n⛔️⛔️⛔️ Oops, invalid row number! Request a row between 0 and', str(self.sample_Size - 1), 'inclusive. ⛔️⛔️⛔️\n')
        