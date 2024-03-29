import sqlite3
import pickle
from taunet import DB_DIR

class TauNetDB:
    def __init__(self):
        print("Connecting to local database...")
        self.connection = sqlite3.connect(f'{DB_DIR}/taunetdb.db')
        self.table = None
        self._create_table_()

    def _create_table_(self):
        pass

    def reset_table(self):
        reset_table_sql = f'DELETE FROM {self.table}'  # SQLite doesn't support TRUNCATE
        self.execute_query(reset_table_sql)

    def remove_table(self):
        remove_sql = f'DROP TABLE {self.table}'
        self.execute_query(remove_sql)

    def execute_query(self, query, data=None):
        try:
            cursor = self.connection.cursor()
            if data:
                cursor.execute(query, data)
            else:
                cursor.execute(query)  # No parameters for queries like table creation
            self.connection.commit()
        except Exception as e:
            print(f"Error executing query: {e}")

    def get_data(self, query, data=None):
        try:
            cursor = self.connection.cursor()
            if data:
                cursor.execute(query, data)
            else:
                cursor.execute(query)  # No parameters
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
                id INTEGER PRIMARY KEY,
                tau TEXT NOT NULL,
                powers BLOB NOT NULL
            );
        '''
        self.execute_query(create_table_sql) 

    def insert_spectra(self, tau, spectra):
        tau_str = str(tau)
        if self.check_tau_exist(tau_str):
            print(f"Spectra with tau {tau_str} already exists.")
            return None

        serialized_spectra = pickle.dumps(spectra)
        insert_sql = 'INSERT INTO spectra (tau, powers) VALUES (?, ?);'  # Using ? as placeholder
        self.execute_query(insert_sql, (tau_str, serialized_spectra))

    def get_spectra(self, tau):
        tau_str = str(tau)
        query = 'SELECT powers FROM spectra WHERE tau = ?;'  # Using ? as placeholder
        result = self.get_data(query, (tau_str,))
        if result:
            return pickle.loads(result[0][0])
        else:
            print(f"No spectra found for tau {tau_str}.")
            return None

    def check_tau_exist(self, tau):
        tau_str = str(tau)
        check_query = 'SELECT COUNT(*) FROM spectra WHERE tau = ?;'  # Using ? as placeholder
        result = self.get_data(check_query, (tau_str,))
        return result and result[0][0] > 0

    def get_all_taus(self):
        query = 'SELECT tau FROM spectra;'
        result = self.get_data(query)
        if result is None:
            print("No data found or there was an error fetching the data.")
            return []
        return [row[0] for row in result]


class MapDB(TauNetDB):
    def __init__(self):
        super().__init__()
        self.tau_table = 'TauTable'
        self.map_table = 'MapTable'
        self._create_table_()

    def _create_table_(self):
        create_tau_table_sql = '''
            CREATE TABLE IF NOT EXISTS TauTable (
                id INTEGER PRIMARY KEY,  -- Auto-increment in SQLite
                tau TEXT UNIQUE NOT NULL
            );
        '''
        self.execute_query(create_tau_table_sql)

        create_map_table_sql = '''
            CREATE TABLE IF NOT EXISTS MapTable (
                id INTEGER PRIMARY KEY,  -- Auto-increment in SQLite
                tau_id TEXT,
                seed INTEGER,
                map TEXT,
                FOREIGN KEY (tau_id) REFERENCES TauTable(id)
            );
        '''
        self.execute_query(create_map_table_sql)

    def insert_map(self, seed, tau, maps):
        tau = str(tau)
        maps = pickle.dumps(maps)
        # Adjustments for SQLite placeholder syntax
        insert_tau_sql = 'INSERT OR IGNORE INTO TauTable (tau) VALUES (?)'
        self.execute_query(insert_tau_sql, (tau,))

        get_tau_id_sql = 'SELECT id FROM TauTable WHERE tau = ?'
        tau_id = self.get_data(get_tau_id_sql, (tau,))[0][0]

        insert_map_sql = 'INSERT INTO MapTable (tau_id, seed, map) VALUES (?, ?, ?)'
        self.execute_query(insert_map_sql, (tau_id, seed, maps))

    def get_map(self, seed, tau):
        tau = str(tau)
        get_map_sql = '''
            SELECT t.tau, m.map
            FROM MapTable m
            JOIN TauTable t ON m.tau_id = t.id
            WHERE m.seed = ?
        '''
        result = self.get_data(get_map_sql, (seed,))

        if result is None or len(result) == 0 or result[0][0] != tau:
            raise ValueError("No matching record found or tau does not match.")
        
        return pickle.loads(result[0][1])

    def check_seed_exist(self, tau, seed):
        tau = str(tau)
        query = '''
            SELECT EXISTS(
                SELECT 1 
                FROM MapTable m
                JOIN TauTable t ON m.tau_id = t.id
                WHERE t.tau = ? AND m.seed = ?
            )
        '''
        result = self.get_data(query, (tau, seed))
        return result[0][0] == 1

    def get_all_tau(self):
        query = 'SELECT tau FROM TauTable'
        result = self.get_data(query)
        return [row[0] for row in result]

    def get_all_seeds(self, tau):
        tau = str(tau)
        query = '''
            SELECT m.seed
            FROM MapTable m
            JOIN TauTable t ON m.tau_id = t.id
            WHERE t.tau = ?
        '''
        result = self.get_data(query, (tau,))
        return [row[0] for row in result]

    def reset_table(self):
        # Reset MapTable
        reset_map_table_sql = f'DELETE FROM {self.map_table};'
        self.execute_query(reset_map_table_sql)
        reset_map_autoincrement_sql = f'UPDATE sqlite_sequence SET seq = 0 WHERE name = "{self.map_table}";'
        self.execute_query(reset_map_autoincrement_sql)
        
        # Reset TauTable
        reset_tau_table_sql = f'DELETE FROM {self.tau_table};'
        self.execute_query(reset_tau_table_sql)
        reset_tau_autoincrement_sql = f'UPDATE sqlite_sequence SET seq = 0 WHERE name = "{self.tau_table}";'
        self.execute_query(reset_tau_autoincrement_sql)
    
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
                id INTEGER PRIMARY KEY,
                model TEXT NOT NULL,
                freq TEXT NOT NULL,
                map BLOB NOT NULL,
                UNIQUE(model, freq)
            );
        '''
        self.execute_query(create_table_sql)

    def insert_map(self, model, freq, map_data):
        model_str = ''.join(model)
        freq_str = str(freq)
        serialized_map = pickle.dumps(map_data)

        insert_sql = 'INSERT INTO FGTable (model, freq, map) VALUES (?, ?, ?)'
        self.execute_query(insert_sql, (model_str, freq_str, serialized_map))

    def get_map(self, model, freq):
        model_str = ''.join(model)
        freq_str = str(freq)

        get_map_sql = 'SELECT map FROM FGTable WHERE model = ? AND freq = ?'
        result = self.get_data(get_map_sql, (model_str, freq_str))

        if result:
            return pickle.loads(result[0][0])
        else:
            raise ValueError("No map found for the given model and frequency.")

    def get_all_freq(self, model):
        model_str = ''.join(model)

        query = 'SELECT freq FROM FGTable WHERE model = ?'
        result = self.get_data(query, (model_str,))
        return [row[0] for row in result]

    def get_all_model(self):
        query = 'SELECT DISTINCT model FROM FGTable'
        result = self.get_data(query)
        return [row[0] for row in result]

    def check_model_exist(self, model, freq):
        model_str = ''.join(model)
        freq_str = str(freq)

        check_query = 'SELECT COUNT(*) FROM FGTable WHERE model = ? AND freq = ?'
        result = self.get_data(check_query, (model_str, freq_str))
        return result and result[0][0] > 0



class NoiseDB(TauNetDB):
    def __init__(self, prefix):
        super().__init__()
        self.table = f'{prefix}NoiseTable'
        self._create_table_()

    def _create_table_(self):
        create_table_sql = f'''
            CREATE TABLE IF NOT EXISTS {self.table} (
                id INTEGER PRIMARY KEY,
                frequency TEXT NOT NULL,
                seed INTEGER NOT NULL,
                map BLOB NOT NULL,
                UNIQUE(frequency, seed)
            );
        '''
        self.execute_query(create_table_sql)

    def insert_noise_data(self, frequency, seed, map_data):
        serialized_map = pickle.dumps(map_data)
        insert_sql = f'INSERT INTO {self.table} (frequency, seed, map) VALUES (?, ?, ?)'
        self.execute_query(insert_sql, (frequency, seed, serialized_map))

    def get_noise_data(self, frequency, seed):
        get_sql = f'SELECT map FROM {self.table} WHERE frequency = ? AND seed = ?'
        result = self.get_data(get_sql, (frequency, seed))

        if result:
            return pickle.loads(result[0][0])
        else:
            raise ValueError("No noise data found for the given frequency and seed.")

    def get_all_frequencies(self):
        query = f'SELECT DISTINCT frequency FROM {self.table}'
        result = self.get_data(query)
        return [row[0] for row in result]

    def check_noise_exist(self, frequency, seed):
        check_query = f'SELECT COUNT(*) FROM {self.table} WHERE frequency = ? AND seed = ?'
        result = self.get_data(check_query, (frequency, seed))
        return result[0][0] > 0
