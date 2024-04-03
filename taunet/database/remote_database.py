import pymysql
import os
import numpy as np
import pickle


class TauNetDB:
    def __init__(self):
        host = 'antolonappan.me'
        user = os.getenv('TAUNET_ROOT_USERNAME')
        password = os.getenv('TAUNET_ROOT_PASSWORD')
        db = 'taunetdb'
        try:
            self.connection = pymysql.connect(host=host, user=user, passwd=password, db=db)
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            raise FileNotFoundError("Database connection failed")
        self.table = None
        self._create_table_()

    def _create_table_(self):
        pass
    
    def reset_table(self):
        reset_table_sql = f'TRUNCATE TABLE {self.table}'
        self.execute_query(reset_table_sql)

    def remove_table(self):
        remove_sql = f'DROP TABLE {self.table}'
        self.execute_query(remove_sql)

    def execute_query(self, query, data=None):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, data)
                self.connection.commit()
        except Exception as e:
            print(f"Error executing query: {e}")

    def get_data(self, query, data=None):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query,data)
                return cursor.fetchall()
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def show_tables(self):
        # Query to fetch table names (MySQL-specific)
        query = "SHOW TABLES"
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                # Fetch all table names and return them
                return [table[0] for table in cursor.fetchall()]
        except Exception as e:
            print(f"Error fetching tables: {e}")
            return None

class SpectrumDB(TauNetDB):
    def __init__(self):
        super().__init__()
        self.table = 'spectra'

    def _create_table_(self):
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS spectra (
                id integer PRIMARY KEY AUTO_INCREMENT,
                tau VARCHAR(255) NOT NULL,
                powers BLOB NOT NULL
            );
        '''
        self.execute_query(create_table_sql)

    def insert_spectra(self, tau, spectra):
        tau_str = str(tau)
        if self.check_tau_exist(tau_str):
            print(f"Spectra with tau {tau_str} already exists.")
            return

        serialized_spectra = pickle.dumps(spectra)
        insert_sql = 'INSERT INTO spectra (tau, powers) VALUES (%s, %s);'
        self.execute_query(insert_sql, (tau_str, serialized_spectra))

    def get_spectra(self, tau):
        tau_str = str(tau)
        query = 'SELECT powers FROM spectra WHERE tau = %s;'
        result = self.get_data(query, (tau_str,))
        if result:
            return pickle.loads(result[0][0])
        else:
            print(f"No spectra found for tau {tau_str}.")
            return None

    def check_tau_exist(self, tau):
        tau_str = str(tau)
        check_query = 'SELECT COUNT(*) FROM spectra WHERE tau = %s;'
        result = self.get_data(check_query, (tau_str,))
        return result and result[0][0] > 0
    
    def get_all_taus(self):
        query = 'SELECT tau FROM spectra;'
        result = self.get_data(query)
        return [float(row[0]) for row in result]


import pickle

class CMBmapDB(TauNetDB):
    def __init__(self, prefix: str, taus: np.ndarray):
        super().__init__()
        self.table = f'{prefix}_CMBmap'
        self._create_table_()
        self.taus = taus
        self.tau_table = f'{prefix}_tau_distribution'
        self._create_tau_table_()

    def _create_table_(self):
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS {} (
                id INT PRIMARY KEY,
                seed INT NOT NULL,
                tau VARCHAR(255) NOT NULL,
                map BLOB NOT NULL
            );
        '''.format(self.table)
        self.execute_query(create_table_sql)
    
    def _create_tau_table_(self):
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS {} (
                id INT PRIMARY KEY AUTO_INCREMENT,
                tau_distribution BLOB NOT NULL
            );
        '''.format(self.tau_table)
        self.execute_query(create_table_sql)

    def insert_tau_distr(self):
        # Serialize the NumPy array
        serialized_taus = pickle.dumps(self.taus)
        insert_sql = 'INSERT INTO {} (tau_distribution) VALUES (%s);'.format(self.tau_table)
        self.execute_query(insert_sql, (serialized_taus,))

    def get_tau_distr(self):
        query = 'SELECT tau_distribution FROM {};'.format(self.tau_table)
        result = self.get_data(query)
        if result:
            # Assuming you want to return the latest entry
            return pickle.loads(result[-1][0])
        else:
            print("No tau distribution data found.")
            return None

    def insert_map(self, map_id, seed, tau, cmb_map):
        if self.check_id_exist(map_id):
            print(f"Map with ID {map_id} already exists.")
            return

        serialized_map = pickle.dumps(cmb_map)
        insert_sql = 'INSERT INTO {} (id, seed, tau, map) VALUES (%s, %s, %s, %s);'.format(self.table)
        self.execute_query(insert_sql, (map_id, seed, tau, serialized_map))

    def check_id_exist(self, map_id):
        check_query = 'SELECT COUNT(*) FROM {} WHERE id = %s;'.format(self.table)
        result = self.get_data(check_query, (map_id,))
        return result and result[0][0] > 0

    def get_seed(self, map_id):
        query = 'SELECT seed FROM {} WHERE id = %s;'.format(self.table)
        result = self.get_data(query, (map_id,))
        if result:
            return result[0][0]
        else:
            print(f"No map found for ID {map_id}.")
            return None

    def get_tau(self, map_id):
        query = 'SELECT tau FROM {} WHERE id = %s;'.format(self.table)
        result = self.get_data(query, (map_id,))
        if result:
            return result[0][0]
        else:
            print(f"No map found for ID {map_id}.")
            return None

    def get_map(self, map_id):
        query = 'SELECT map FROM {} WHERE id = %s;'.format(self.table)
        result = self.get_data(query, (map_id,))
        if result:
            return pickle.loads(result[0][0])
        else:
            print(f"No map found for ID {map_id}.")
            return None



class ForegroundDB(TauNetDB):
    def __init__(self):
        super().__init__()
        self.table = 'FGTable'
        self._create_table_()

    def _create_table_(self):
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS FGTable (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model VARCHAR(255) NOT NULL,
                freq VARCHAR(255) NOT NULL,
                map BLOB NOT NULL,
                UNIQUE(model, freq)
            );
        '''
        self.execute_query(create_table_sql)

    def insert_map(self, model, freq, map_data):
        model_str = ''.join(model)
        freq_str = str(freq)
        serialized_map = pickle.dumps(map_data)

        insert_sql = 'INSERT INTO FGTable (model, freq, map) VALUES (%s, %s, %s)'
        self.execute_query(insert_sql, (model_str, freq_str, serialized_map))

    def get_map(self, model, freq):
        model_str = ''.join(model)
        freq_str = str(freq)

        get_map_sql = 'SELECT map FROM FGTable WHERE model = %s AND freq = %s'
        result = self.get_data(get_map_sql, (model_str, freq_str))

        if result:
            return pickle.loads(result[0][0])
        else:
            raise ValueError("No map found for the given model and frequency.")

    def get_all_freq(self, model):
        model_str = ''.join(model)

        query = 'SELECT freq FROM FGTable WHERE model = %s'
        result = self.get_data(query, (model_str,))
        return [row[0] for row in result]

    def get_all_model(self):
        query = 'SELECT DISTINCT model FROM FGTable'
        result = self.get_data(query)
        return [row[0] for row in result]

    def check_model_exist(self, model, freq):
        model_str = ''.join(model)
        freq_str = str(freq)

        check_query = 'SELECT COUNT(*) FROM FGTable WHERE model = %s AND freq = %s'
        result = self.get_data(check_query, (model_str, freq_str))
        return result[0][0] > 0
    

class NoiseDB(TauNetDB):
    def __init__(self, prefix):
        super().__init__()
        self.table = f'{prefix}NoiseTable'
        self._create_table_()

    def _create_table_(self):
        create_table_sql = f'''
            CREATE TABLE IF NOT EXISTS {self.table} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                frequency VARCHAR(255) NOT NULL,
                seed INT NOT NULL,
                map BLOB NOT NULL,
                UNIQUE(frequency, seed)
            );
        '''
        self.execute_query(create_table_sql)

    def insert_noise(self, frequency, seed, map_data):
        serialized_map = pickle.dumps(map_data)
        insert_sql = f'INSERT INTO {self.table} (frequency, seed, map) VALUES (%s, %s, %s)'
        self.execute_query(insert_sql, (frequency, seed, serialized_map))

    def get_noise(self, frequency, seed):
        get_sql = f'SELECT map FROM {self.table} WHERE frequency = %s AND seed = %s'
        result = self.get_data(get_sql, (frequency, seed))

        if result:
            return pickle.loads(result[0][0])
        else:
            raise ValueError("No noise data found for the given frequency and seed.")

    def get_all_frequencies(self):
        query = f'SELECT DISTINCT frequency FROM {self.table}'
        result = self.get_data(query)
        return [row[0] for row in result]
    
    def get_all_seeds(self, freq):
        query = f'SELECT DISTINCT seed FROM {self.table} WHERE frequency = %s'
        result = self.get_data(query, (freq,))
        return [row[0] for row in result] if result else []

    def check_noise_exist(self, frequency, seed):
        check_query = f'SELECT COUNT(*) FROM {self.table} WHERE frequency = %s AND seed = %s'
        result = self.get_data(check_query, (frequency, seed))
        return result[0][0] > 0

