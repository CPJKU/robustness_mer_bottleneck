import csv


class Logger:
    """ Simple logger class that is used to log adversarial attacks into csv files. Columns can be custom. """
    def __init__(self, log_file_path, columns=None):
        """
        Parameters
        ----------
        log_file_path: str
            Path + filename of csv file where log should be stored.
        columns : list, optional
            List of str, defining column headers (if None, uses [epoch, train loss, train accuracy]).
        """
        self.log_path = log_file_path
        if columns is None:
            self.columns = ['epoch', 'train loss', 'train accuracy']
        else:
            self.columns = columns

        with open(self.log_path, 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(columns)

    def append(self, value_list):
        """
        Method that writes one line to csv-log.

        Parameters
        ----------
        value_list : list
            List of elements (strings) that should be written to log file.
        """
        with open(self.log_path, 'a') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(value_list)
