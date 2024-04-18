import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base class for reading index name


class ReadIndexName:

    def __init__(self, name=None):
        self.name = name

# Class for reading URL with index name


class ReadURL(ReadIndexName):

    def __init__(self, name=None, url=None):
        super().__init__(name)
        self.url = url


class ReadTable(ReadURL):
    """
    Class for reading a table from a URL.
    Inherits from ReadURL class.
    """

    def __init__(self, name=None, url=None):
        """
        Initializes the ReadTable object with name and URL.

        Args:
        - name (str): Name of the object.
        - url (str): URL from which the data will be scraped.
        """
        super().__init__(name, url)

    def read_table(self, t1=None, t2=None, t3=None, t4=None, t5=None, t6=None, t7=None, t8=None, t9=None):
        """
        Method to read table data from the URL.

        Args:
        - t1 (str): Tag name for finding the table division.
        - t2 (str): Attribute name for finding the table division.
        - t3 (str): Attribute value for finding the table division.
        - t4 (str): Tag name for finding the table.
        - t5 (str): Attribute name for finding the table.
        - t6 (str): Attribute value for finding the table.
        - t7 (str): Tag name for finding the table rows.
        - t8 (str): Tag name for finding table headers.
        - t9 (str): Tag name for finding table data cells.

        Returns:
        - DataFrame: Table data as DataFrame.
        """
        try:
            # Fetching the webpage content
            response = requests.get(self.url)
            if response.status_code == 200:
                # Parsing the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                # Finding the table division
                table_div = soup.find(t1, {t2: t3})
                if table_div:
                    # Finding the table
                    table = table_div.find(t4, {t5: t6})
                    if table:
                        # Extracting rows from the table
                        rows = table.find_all(t7)
                        table_element = []
                        for row in rows:
                            # Extracting columns from each row
                            columns = row.find_all([t8, t9])
                            row_data = [col.get_text(strip=True).replace(
                                ',', '') for col in columns]
                            table_element.append(row_data)
                        # Creating DataFrame from the table data
                        return pd.DataFrame(table_element[1:], columns=table_element[0])
                    else:
                        return None
                else:
                    return None
            else:
                return None
        except Exception as e:
            return e


class FinancialSummary(ReadTable):
    """
    Class for reading balance sheet data from a webpage.
    Inherits from ReadTable class.
    """

    def __init__(self, name=None, url=None):
        """
        Initializes the ReadBalanceSheet object with name and URL.

        Args:
        - name (str): Name of the object.
        - url (str): URL from which the data will be scraped.
        """
        super().__init__(name, url)

    def scrape_company_information_list(self):
        """
        Method to scrape company information from the webpage.

        Returns:
        - dict: Dictionary containing company information.
        """
        try:
            # Fetching the webpage content
            response = requests.get(self.url)
            if response.status_code == 200:
                # Parsing the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                # Finding the company information list
                info_list = soup.find('ul', {'id': 'top-ratios'})
                if info_list:
                    # Extracting company information items
                    info_items = info_list.find_all('li')
                    company_info = {}
                    for item in info_items:
                        label_span = item.find('span', {'class': 'name'})
                        value_span = item.find('span', {'class': 'value'})
                        label = label_span.get_text(
                            strip=True) if label_span else None
                        value = value_span.get_text(
                            strip=True) if value_span else None
                        # Cleaning the value (removing currency symbols and commas)
                        value = value.replace(u'\u20B9', "").replace(
                            ",", "").replace("Cr", "")
                        company_info[label] = value
                    return company_info
                else:
                    return None
            else:
                return None
        except Exception as e:
            return e

    def get_balance_sheet_data_df(self):
        """
        Method to get balance sheet data as DataFrame.

        Returns:
        - DataFrame: Balance sheet data.
        """
        try:
            # Reading balance sheet data
            return self.read_table(t1='section', t2='id', t3='balance-sheet',
                                   t4='table', t5='class', t6='data-table',
                                   t7='tr', t8='th', t9='td')
        except Exception as e:
            return e

    def get_profit_loss(self):
        """
        Method to get profit loss data as DataFrame.

        Returns:
        - DataFrame: Profit loss data.
        """
        try:
            # Reading profit loss data
            return self.read_table(t1='section', t2='id', t3='profit-loss',
                                   t4='table', t5='class', t6='data-table',
                                   t7='tr', t8='th', t9='td')
        except Exception as e:
            return e

    def get_quaterly_data_df(self):
        """
        Method to get quarterly data as DataFrame.

        Returns:
        - DataFrame: Quarterly data.
        """
        try:
            # Reading quarterly data
            return self.read_table(t1='section', t2='id', t3='quarters',
                                   t4='table', t5='class', t6='data-table',
                                   t7='tr', t8='th', t9='td')
        except Exception as e:
            return e

    def get_cash_flow_df(self):
        """
        Get the cash flow DataFrame.

        Returns:
            DataFrame: The cash flow DataFrame.
        """
        try:
            # Read the table with specified parameters
            return self.read_table(t1='section', t2='id', t3='cash-flow',
                                t4='table', t5='class', t6='data-table',
                                t7='tr', t8='th', t9='td')
        except Exception as e:
            return e



class CompanyInformation(FinancialSummary):
    """
    Class for storing company information obtained from a webpage.
    Inherits from FinancialSummary class.
    """

    def __init__(self, name=None, url=None):
        """
        Initializes the CompanyInformation object with name and URL.

        Args:
        - name (str): Name of the object.
        - url (str): URL from which the data will be scraped.
        """
        super().__init__(name, url)

    def company_info(self):
        """
        Method to retrieve company information and convert it to DataFrame.

        Returns:
        - DataFrame: Company information as DataFrame.
        """
        try:
            # Scrape company information and convert it to DataFrame
            return pd.DataFrame.from_dict(self.scrape_company_information_list(), orient='index').T
        except Exception as e:
            return e


class ReadCompanyBalanceSheet(FinancialSummary):
    """
    Class for reading company balance sheet data from a webpage.
    Inherits from FinancialSummary class.
    """

    def __init__(self, name=None, url=None):
        """
        Initializes the ReadCompanyBalanceSheet object with name and URL.

        Args:
        - name (str): Name of the object.
        - url (str): URL from which the data will be scraped.
        """
        super().__init__(name, url)

    def read_company_bs(self):
        """
        Method to read company balance sheet data from the webpage.

        Returns:
        - DataFrame: Company balance sheet data.
        """
        try:
            return self.get_balance_sheet_data_df()
        except Exception as e:
            return e

    def balance_sheet_df(self):
        """
        Method to convert balance sheet data to DataFrame.

        Returns:
        - DataFrame: Company balance sheet data as DataFrame.
        """
        try:
            # Reading company balance sheet data
            df = self.read_company_bs().set_index('')
            # Converting columns to numeric except for those containing '%'
            return df.T.applymap(lambda x: pd.to_numeric(x, errors='coerce') if '%' not in str(x) else x)
        except Exception as e:
            return e


class ReadProfitLoss(FinancialSummary):
    """
    Class for reading profit loss data from a webpage.
    Inherits from FinancialSummary class.
    """

    def __init__(self, name=None, url=None):
        """
        Initializes the ReadProfitLoss object with name and URL.

        Args:
        - name (str): Name of the object.
        - url (str): URL from which the data will be scraped.
        """
        super().__init__(name, url)

    def read_profit_loss(self):
        """
        Method to read profit loss data from the webpage.

        Returns:
        - DataFrame: Profit loss data.
        """
        try:
            return self.get_profit_loss()
        except Exception as e:
            return e

    def profit_loss_df(self):
        """
        Method to convert profit loss data to DataFrame.

        Returns:
        - DataFrame: Profit loss data as DataFrame.
        """
        try:
            # Reading profit loss data
            df = self.read_profit_loss().set_index('')
            # Converting columns to numeric except for those containing '%'
            return df.T.applymap(lambda x: pd.to_numeric(x, errors='coerce') if '%' not in str(x) else x)
        except Exception as e:
            return e


class ReadQuaterlyResult(FinancialSummary):
    """
    Class to read quarterly result data from a webpage.
    Inherits from FinancialSummary class.
    """

    def __init__(self, name=None, url=None):
        """
        Initializes the ReadQuaterlyResult object with name and URL.

        Args:
        - name (str): Name of the object.
        - url (str): URL from which the data will be scraped.
        """
        super().__init__(name, url)

    def read_bs(self):
        """
        Method to read quarterly result data from the webpage.

        Returns:
        - DataFrame: Quarterly result data.
        """
        try:
            return self.get_quaterly_data_df()
        except Exception as e:
            return e

    def quaterly_result_df(self):
        """
        Method to convert quarterly result data to DataFrame.

        Returns:
        - DataFrame: Quarterly result data as DataFrame.
        """
        try:
            # Reading quarterly result data
            df = self.read_bs().set_index('')
            # Converting columns to numeric except for those containing '%'
            return df.T.applymap(lambda x: pd.to_numeric(x, errors='coerce') if '%' not in str(x) else x)
        except Exception as e:
            return e


class CashFlowStatements(FinancialSummary):
    """
    Class to handle cash flow statements.
    Inherits from FinancialSummary.
    """

    def __init__(self, name=None, url=None):
        """
        Initialize the CashFlowStatements object.

        Parameters:
            name (str): The name of the financial entity.
            url (str): The URL of the financial data.
        """
        super().__init__(name, url)

    def read_cash_flow_table(self):
        """
        Read the cash flow table.

        Returns:
            DataFrame: The cash flow table DataFrame.
        """
        try:
            return self.get_cash_flow_df()
        except Exception as e:
            return e

    def cash_flow_df(self):
        """
        Process the cash flow DataFrame.

        Returns:
            DataFrame: The processed cash flow DataFrame.
        """
        try:
            df = self.read_cash_flow_table().set_index('')
            # Convert numeric values to numeric type, preserving percentage values
            return df.T.applymap(lambda x: pd.to_numeric(x, errors='coerce') if '%' not in str(x) else x)
        except Exception as e:
            return e



class GetTickerName:
    """
    Base class to store the ticker name and URL.
    """

    def __init__(self, ticker_name, url):
        """
        Initialize the GetTickerName object.

        Parameters:
            ticker_name (str): The ticker symbol of the company.
            url (str): The URL of the company's financial data.
        """
        self.name = ticker_name  # Store the ticker name
        self.url = url  # Store the URL


class GetBalanceSheet(GetTickerName):
    """
    Class to retrieve and process balance sheet data.
    Inherits from GetTickerName.
    """

    def __init__(self, ticker_name, url):
        """
        Initialize the GetBalanceSheet object.

        Parameters:
            ticker_name (str): The ticker symbol of the company.
            url (str): The URL of the company's financial data.
        """
        super().__init__(ticker_name, url)

    def balance_sheet_data(self):
        """
        Retrieve balance sheet data and save it to a file.
        """
        try:
            # Instantiate ReadCompanyBalanceSheet
            re = ReadCompanyBalanceSheet(name=self.name, url=self.url)
            # Convert balance sheet data to CSV format and save to file
            re.balance_sheet_df().T.to_csv(self.name + '_balance_sheet.txt', sep='\t', index=True)
        except Exception as e:
            return e  # Return any exceptions encountered


class GetQuaterlyResults(GetTickerName):
    """
    Class to retrieve and process quarterly results data.
    Inherits from GetTickerName.
    """

    def __init__(self, ticker_name, url):
        """
        Initialize the GetQuaterlyResults object.

        Parameters:
            ticker_name (str): The ticker symbol of the company.
            url (str): The URL of the company's financial data.
        """
        super().__init__(ticker_name, url)

    def quaterly_sheet_data(self):
        """
        Retrieve quarterly results data and save it to a file.
        """
        try:
            # Instantiate ReadQuaterlyResult
            re = ReadQuaterlyResult(name=self.name, url=self.url)
            # Convert quarterly results data to CSV format and save to file
            re.quaterly_result_df().T.to_csv(self.name + '.txt', sep='\t', index=True)
        except Exception as e:
            return e  # Return any exceptions encountered


class GetProfitAndLossData(GetTickerName):
    """
    Class to retrieve and process profit and loss data.
    Inherits from GetTickerName.
    """

    def __init__(self, ticker_name, url):
        """
        Initialize the GetProfitAndLossData object.

        Parameters:
            ticker_name (str): The ticker symbol of the company.
            url (str): The URL of the company's financial data.
        """
        super().__init__(ticker_name, url)

    def profit_loss_data(self):
        """
        Retrieve profit and loss data and save it to a file.
        """
        try:
            # Instantiate ReadProfitLoss
            pf = ReadProfitLoss(name=self.name, url=self.url)
            # Convert profit and loss data to CSV format and save to file
            pf.profit_loss_df().T.to_csv(self.name + '_proft_loss.txt', sep='\t', index=True)
        except Exception as e:
            return e  # Return any exceptions encountered

class GetCashFlowData(GetTickerName):
    """
    Class to retrieve and process cash flow data.
    Inherits from GetTickerName.
    """

    def __init__(self, ticker_name, url):
        """
        Initialize the GetCashFlowData object.

        Parameters:
            ticker_name (str): The ticker symbol of the company.
            url (str): The URL of the company's financial data.
        """
        super().__init__(ticker_name, url)

    def company_cash_flow(self):
        """
        Retrieve and process company cash flow data.
        """
        try:
            cash = CashFlowStatements(name=self.name, url=self.url)  # Instantiate CashFlowStatements
            # Convert cash flow data to CSV format and save to file
            cash.cash_flow_df().T.to_csv(self.name + '_cash_flow.txt', sep='\t', index=True)
        except Exception as e:
            return e



if __name__ == '__main__':

    company_name = 'INFY'
    url = f'https://www.screener.in/company/{company_name}/consolidated/'

    balance = GetBalanceSheet(ticker_name=company_name, url=url)
    balance.balance_sheet_data()

    quater = GetQuaterlyResults(ticker_name=company_name, url=url)
    quater.quaterly_sheet_data()

    profitloss = GetProfitAndLossData(ticker_name=company_name, url=url)
    profitloss.profit_loss_data()

    cash_data = GetCashFlowData(ticker_name=company_name, url=url)
    cash_data.company_cash_flow()
