import pymysql
import os
import pickle


class TauNetDB:
    def __init__(self):
        print("Connecting to the remote database...")
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


class MapDB(TauNetDB):
    def __init__(self):
        super().__init__()
        self.tau_table = 'TauTable'
        self.map_table = 'MapTable'
        self._create_table_()

    def _create_table_(self):
        create_tau_table_sql = '''
            CREATE TABLE IF NOT EXISTS TauTable (
                id INT AUTO_INCREMENT PRIMARY KEY,
                tau VARCHAR(255) UNIQUE NOT NULL
            );
        '''
        self.execute_query(create_tau_table_sql)

        create_map_table_sql = '''
            CREATE TABLE IF NOT EXISTS MapTable (
                id INT AUTO_INCREMENT PRIMARY KEY,
                tau_id INT,
                seed INT,
                map BLOB,
                FOREIGN KEY (tau_id) REFERENCES TauTable(id)
            );
        '''
        self.execute_query(create_map_table_sql)

    def insert_map(self, seed, tau, maps):
        tau_str = str(tau)
        serialized_maps = pickle.dumps(maps)

        insert_tau_sql = 'INSERT IGNORE INTO TauTable (tau) VALUES (%s)'
        self.execute_query(insert_tau_sql, (tau_str,))

        get_tau_id_sql = 'SELECT id FROM TauTable WHERE tau = %s'
        tau_id = self.get_data(get_tau_id_sql, (tau_str,))[0][0]

        insert_map_sql = 'INSERT INTO MapTable (tau_id, seed, map) VALUES (%s, %s, %s)'
        self.execute_query(insert_map_sql, (tau_id, seed, serialized_maps))

    def get_map(self, seed, tau):
        tau_str = str(tau)
        get_map_sql = '''
            SELECT t.tau, m.map
            FROM MapTable m
            JOIN TauTable t ON m.tau_id = t.id
            WHERE m.seed = %s
        '''
        result = self.get_data(get_map_sql, (seed,))

        if result is None or len(result) == 0 or result[0][0] != tau_str:
            raise ValueError("No matching record found or tau does not match.")
        
        return pickle.loads(result[0][1])


    def check_seed_exist(self, tau, seed):
        tau_str = str(tau)
        query = '''
            SELECT EXISTS(
                SELECT 1 
                FROM MapTable m
                JOIN TauTable t ON m.tau_id = t.id
                WHERE t.tau = %s AND m.seed = %s
            )
        '''
        result = self.get_data(query, (tau_str, seed))
        return result[0][0] == 1

    def get_all_seeds(self, tau):
        tau_str = str(tau)
        query = '''
            SELECT m.seed
            FROM MapTable m
            JOIN TauTable t ON m.tau_id = t.id
            WHERE t.tau = %s
        '''
        result = self.get_data(query, (tau_str,))
        return [row[0] for row in result]
    
    def get_all_tau(self):
        query = 'SELECT tau FROM TauTable'
        result = self.get_data(query)
        return [row[0] for row in result]

    def reset_table(self):
        # In MySQL, resetting a table typically means emptying it.
        # Auto-increment reset is done differently compared to SQLite
        # Reset MapTable
        reset_map_table_sql = f'TRUNCATE TABLE {self.map_table};'
        self.execute_query(reset_map_table_sql)
        
        # Reset TauTable
        reset_tau_table_sql = f'TRUNCATE TABLE {self.tau_table};'
        self.execute_query(reset_tau_table_sql)
    
    def remove_table(self):
        # Remove MapTable first due to foreign key constraint
        remove_map_table_sql = f'DROP TABLE IF EXISTS {self.map_table};'
        self.execute_query(remove_map_table_sql)
        
        # Then remove TauTable
        remove_tau_table_sql = f'DROP TABLE IF EXISTS {self.tau_table};'
        self.execute_query(remove_tau_table_sql)


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

    def __init__(self,prefix):
        super().__init__()
        self.table = f'{prefix}NoiseTable'
        self.prefix = prefix
        self._create_table_()





