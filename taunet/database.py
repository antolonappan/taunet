import sqlite3
import json
import numpy as np

class CMBObject:
    __slots__ = ['tau', 'map']
    def __init__(self, tau, maps):
        self.tau = tau
        self.map = maps


class NoiseObject:
    __slots__ = ['frequency', 'map']
    def __init__(self, frequency, maps):
        self.frequency = frequency
        self.map = maps

class CleanedCMBObject:
    __slots__ = ['tau', 'frequency', 'map']
    def __init__(self, tau, frequency, maps):
        self.tau = tau
        self.frequency = frequency
        self.map = maps

class FGObject:
    __slots__ = ['model', 'frequency', 'map']
    def __init__(self, model, frequency, maps):
        self.model = model
        self.frequency = frequency
        self.map  = maps


class TauNetDB:
    def __init__(self, db_file='taunet.db'):
        self.conn = sqlite3.connect(db_file)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS CMB (
                            cmb_index INTEGER PRIMARY KEY,
                            tau REAL,
                            map TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS Noise (
                            noise_index INTEGER,
                            frequency INTEGER,
                            map TEXT,
                            PRIMARY KEY (noise_index, frequency))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS CleanedCMB (
                            cmb_index INTEGER PRIMARY KEY,
                            tau REAL,
                            frequency INTEGER,
                            map TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS FG (
                            model TEXT,
                            frequency INTEGER,
                            map TEXT,
                            PRIMARY KEY (model, frequency))''')
        self.conn.commit()

        self.conn.commit()

    def _serialize_map(self, map):
        if isinstance(map, np.ndarray):
            map = map.tolist()
        return json.dumps(map)

    def insert_cmb(self, cmb_index, tau, map):
        try:
            cursor = self.conn.cursor()
            map_json = self._serialize_map(map)
            cursor.execute('INSERT INTO CMB (cmb_index, tau, map) VALUES (?, ?, ?)', (cmb_index, tau, map_json))
            self.conn.commit()
        except sqlite3.IntegrityError:
            print(f"CMB record with index {cmb_index} already exists.")

    def edit_cmb(self, cmb_index, tau, map):
        cursor = self.conn.cursor()
        map_json = self._serialize_map(map)
        cursor.execute('UPDATE CMB SET tau = ?, map = ? WHERE cmb_index = ?', (tau, map_json, cmb_index))
        self.conn.commit()

    def insert_noise(self, noise_index, frequency, map):
        try:
            cursor = self.conn.cursor()
            map_json = self._serialize_map(map)
            cursor.execute('INSERT INTO Noise (noise_index, frequency, map) VALUES (?, ?, ?)', (noise_index, frequency, map_json))
            self.conn.commit()
        except sqlite3.IntegrityError:
            print(f"Noise record with index {noise_index} and frequency {frequency} already exists.")

    def edit_noise(self, noise_index, frequency, map):
        cursor = self.conn.cursor()
        map_json = self._serialize_map(map)
        cursor.execute('UPDATE Noise SET map = ? WHERE noise_index = ? AND frequency = ?', (map_json, noise_index, frequency))
        if cursor.rowcount == 0:
            print(f"No existing noise record found for index {noise_index} and frequency {frequency}.")
        else:
            self.conn.commit()  

    def _deserialize_map(self, map_json):
        map_list = json.loads(map_json)
        return np.array(map_list)

    def get_cmb(self, cmb_index):
        cursor = self.conn.cursor()
        cursor.execute('SELECT tau, map FROM CMB WHERE cmb_index = ?', (cmb_index,))
        row = cursor.fetchone()
        if row:
            map_array = self._deserialize_map(row[1])
            return CMBObject(row[0], map_array)
        else:
            return None

    def get_noise(self, noise_index, frequency):
        cursor = self.conn.cursor()
        cursor.execute('SELECT map FROM Noise WHERE noise_index = ? AND frequency = ?', (noise_index, frequency))
        row = cursor.fetchone()
        if row:
            map_array = self._deserialize_map(row[0])
            return NoiseObject( frequency, map_array)
        else:
            return None
        
    def insert_cleaned_cmb(self, cmb_index, tau, frequency, map):
        try:
            cursor = self.conn.cursor()
            map_json = self._serialize_map(map)
            cursor.execute('INSERT INTO CleanedCMB (cmb_index, tau, frequency, map) VALUES (?, ?, ?, ?)', 
                           (cmb_index, tau, frequency, map_json))
            self.conn.commit()
        except sqlite3.IntegrityError:
            print(f"Cleaned CMB record with index {cmb_index} already exists.")

    def edit_cleaned_cmb(self, cmb_index, tau, frequency, map):
        cursor = self.conn.cursor()
        map_json = self._serialize_map(map)
        cursor.execute('UPDATE CleanedCMB SET tau = ?, frequency = ?, map = ? WHERE cmb_index = ?', 
                       (tau, frequency, map_json, cmb_index))
        self.conn.commit()

    def get_cleaned_cmb(self, cmb_index):
        cursor = self.conn.cursor()
        cursor.execute('SELECT tau, frequency, map FROM CleanedCMB WHERE cmb_index = ?', (cmb_index,))
        row = cursor.fetchone()
        if row:
            map_array = self._deserialize_map(row[2])
            return CleanedCMBObject(row[0], row[1], map_array)
        else:
            return None
    
    def insert_fg(self, model, frequency, map):
        try:
            cursor = self.conn.cursor()
            map_json = self._serialize_map(map)
            cursor.execute('INSERT INTO FG (model, frequency, map) VALUES (?, ?, ?)',
                           (model, frequency, map_json))
            self.conn.commit()
        except sqlite3.IntegrityError:
            print(f"FG record with model {model} and frequency {frequency} already exists.")

    def edit_fg(self, model, frequency, map):
        cursor = self.conn.cursor()
        map_json = self._serialize_map(map)
        cursor.execute('UPDATE FG SET map = ? WHERE model = ? AND frequency = ?',
                       (map_json, model, frequency))
        self.conn.commit()

    def get_fg(self, model, frequency):
        cursor = self.conn.cursor()
        cursor.execute('SELECT map FROM FG WHERE model = ? AND frequency = ?', (model, frequency))
        row = cursor.fetchone()
        if row:
            map_array = self._deserialize_map(row[0])
            return FGObject(model, frequency, map_array)
        else:
            return None
        
    def total_map(self,index,frequency,add_noise=True,add_fg=['s0','d0']):
        maps = self.get_cmb(index).map
        if add_noise:
            maps += self.get_noise(index, frequency).map
        if add_fg:
            for fg in add_fg:
                maps += self.get_fg(fg, frequency).map
        return maps