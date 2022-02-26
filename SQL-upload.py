# %%
import pyodbc
import pandas as pd
from extractor.DataGenerator import *
import mysql.connector
# from MySQLdb import _mysql
# db=_mysql.connect()
# %%
# insert data from csv file into dataframe.
# working directory for csv file: type "pwd" in Azure Data Studio or Linux
# working directory in Windows c:\users\username
#! Check OS to change SymLink usage
# profile : str = 'FUDS'
Data    : str = 'Data/'
# dataGenerator = DataGenerator(train_dir=f'{Data}A123_Matt_Set',
#                               valid_dir=f'{Data}A123_Matt_Val',
#                               test_dir=f'{Data}A123_Matt_Test',
#                               columns=[
#                                 'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
#                                 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
#                                 ],
#                               PROFILE_range = profile)

# Some other example server values are
# server = 'localhost\sqlexpress' # for a named instance
# server = 'myserver,port' # to specify an alternate port
# %%
server = '192.168.1.254' 
database = 'battery_data' 
username = 'TF' 
password = 'tensorflow' 
# cnxn = db.connect(
#     # 'DRIVER={SQL Server};'
#     f'SERVER={server};DATABASE={database};UID={username};PWD={password}'
#     )
# sqlEngine = create_engine(
#     f'mysql+pymysql://{username}:{server}/test',
#      pool_recycle=3600)
# cursor = cnxn.cursor()
db_conn = mysql.connector.connect(
      host=server,
      user=username,
      passwd=password,
      database=database
    )
if db_conn.is_connected():
    info = db_conn.get_server_info()
    #print("Connected to MySQL Server version ", db_Info)
    cursor = db_conn.cursor()
    #cursor.execute("select database();")
    cursor.execute("show tables;")
    #record = cursor.fetchone()
    record = cursor.fetchall()
    tables = []
    for element in record:
        tables.append(element[0])    
    print(tables)
else:
    print("Error in the connection")
# %%
# Insert Dataframe into SQL Server:
df : pd.DataFrame = pd.read_excel(io=f'{Data}A123_Matt_Test/A1-007-DST-US06-FUDS-25-20120827.xlsx',
                    sheet_name=1,
                    header=0, names=None, index_col=None,
                    usecols=['Step_Index'] + [
                                'Current(A)', 'Voltage(V)', 'Temperature (C)_1',
                                'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'
                                ],
                    squeeze=False,
                    dtype=np.float32,
                    engine='openpyxl', converters=None, true_values=None,
                    false_values=None, skiprows=None, nrows=None,
                    na_values=None, keep_default_na=True, na_filter=True,
                    verbose=False, parse_dates=False, date_parser=None,
                    thousands=None, comment=None, skipfooter=0,
                    convert_float=True, mangle_dupe_cols=True
                )
cursor.execute(
    "DROP TABLE `{database}`.`test`;"
)
cursor.execute(
        f"CREATE TABLE `{database}`.`test` ("
        "`index` INT NOT NULL,"
        f"`{df.columns[0]}` VARCHAR(45) NULL,"
        f"`{df.columns[1]}` VARCHAR(45) NULL,"
        f"`{df.columns[2]}` VARCHAR(45) NULL,"
        f"`{df.columns[3]}` VARCHAR(45) NULL,"
        f"`{df.columns[4]}` VARCHAR(45) NULL,"
        f"`{df.columns[5]}` VARCHAR(45) NULL,"
        "PRIMARY KEY (`index`));"
    )
# df.to_sql('test2', con=db_conn, schema=database)
for index, row in df.iterrows():
     cursor.execute(
         f"INSERT INTO `{database}`.`test` (`index`,`{df.columns[0]}`,`{df.columns[1]}`,`{df.columns[2]}`,`{df.columns[3]}`,`{df.columns[4]}`, `{df.columns[5]}`) "
         f"values('{index}', {row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}, {row[5]});",         
         )
db_conn.commit()
# %%
cursor.execute("SELECT * FROM battery_data.test;")
#record = cursor.fetchone()
record = cursor.fetchall()
tables = []
for element in record[:5]:
    print(element)
# %%
cursor.close()
db_conn.close()
print("MySQL connection is closed")
# %%