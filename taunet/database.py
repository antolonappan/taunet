import pymysql
import os
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

class CMBmapDB(TauNetDB):
    def __init__(self, pre_table):
        super().__init__()
        self.table = f'{pre_table}_cmb'
        self.create_table()

    def create_table(self):
        create_table_sql = f'''
            CREATE TABLE IF NOT EXISTS {self.table} (
                id integer PRIMARY KEY NOT NULL,
                tau VARCHAR(255) NOT NULL,
                cmb_map BLOB NOT NULL
            );
        '''
        self.execute_query(create_table_sql)

    def insert_map(self, seed, tau, maps):
        if self.check_seed_exist(seed):
            print(f"Map with seed {seed} already exists.")
            return

        str_tau = str(tau)
        pickled_maps = pickle.dumps(maps)
        insert_sql = f'INSERT INTO {self.table} (id, tau, cmb_map) VALUES (%s, %s, %s);'
        self.execute_query(insert_sql, (seed, str_tau, pickled_maps))
    
    def get_map(self, seed):
        get_sql = f'SELECT tau, cmb_map FROM {self.table} WHERE id = %s;'
        data = self.get_data(get_sql, (seed,))
        if len(data) == 0:
            print(f"No map found for seed {seed}.")
            return None
        tau = data[0][0]
        maps = pickle.loads(data[0][1])
        return tau, maps
    
    def check_seed_exist(self, seed):
        check_sql = f'SELECT COUNT(*) FROM {self.table} WHERE id = %s;'
        data = self.get_data(check_sql, (seed,))
        return data and data[0][0] > 0