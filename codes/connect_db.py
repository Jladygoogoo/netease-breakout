import pymysql

class MyConn:
	def __init__(self):
		self.conn = pymysql.connect(host="127.0.0.1", port=3306, user="root",
							db="NetEase_proxied", password="SFpqwnj285798,.")
		self.cache_count = 0

	def query(self, targets=None, conditions=None, sql=None, table="tracks", fetchall=True):
		if not sql:
			sql = "SELECT {} FROM {}".format(','.join(targets), table)
			if conditions:
				sql += " WHERE"
				for i, k in enumerate(conditions):
					sql += " {}=%s".format(k)
					if i<len(conditions)-1:
						sql += " AND"

		with self.conn.cursor() as cursor:
			if conditions:
				cursor.execute(sql, tuple(conditions.values()))
			else:
				cursor.execute(sql)

		if fetchall:
			return cursor.fetchall()
		return cursor.fetchone()


	def update(self, settings=None, conditions=None, sql=None, table="tracks"):
		if not sql:
			sql = "UPDATE {} SET".format(table)
			for i, k in enumerate(settings):
				sql += " {}=%s".format(k)
				if i<len(settings)-1:
					sql += ","
			if conditions:
				sql += " WHERE"
				for i, k in enumerate(conditions):
					sql += " {}=%s".format(k)
					if i<len(conditions)-1:
						sql += " AND"

		args = None
		if settings:
			args = tuple(settings.values())
		if conditions:
			args += tuple(conditions.values())

		with self.conn.cursor() as cursor:
				cursor.execute(sql, args=args)

		self.conn.commit()


	def insert(self, settings=None, sql=None, table="tracks"):
		if not sql:
			sql = "INSERT INTO {} SET".format(table)
			for i, k in enumerate(settings):
				sql += " {}=%s".format(k)
				if i<len(settings)-1:
					sql += ","

		with self.conn.cursor() as cursor:
			cursor.execute(sql, tuple(settings.values()))

		self.conn.commit()


	def insert_or_update(self, settings=None, sql=None, table="tracks"):
		'''
		不存在则insert，存在则update。settings中包含键值
		'''
		if not sql:
			sql = "REPLACE INTO {} SET".format(table)
			for i, k in enumerate(settings):
				sql += " {}=%s".format(k)
				if i<len(settings)-1:
					sql += ","

		with self.conn.cursor() as cursor:
			cursor.execute(sql, tuple(settings.values()))

		self.conn.commit()


	def delete(self, table="tracks", conditions=None, sql=None):
		if not sql and not conditions:
			print("Operation forbidden.")
			return
		if not sql:
			sql = "DELETE FROM {}".format(table)
			if conditions:
				sql += " WHERE"
				for i, k in enumerate(conditions):
					sql += " {}=%s".format(k)
					if i<len(conditions)-1:
						sql += " AND"				
		if conditions:
			args = tuple(conditions.values())
		with self.conn.cursor() as cursor:
			cursor.execute(sql, args=args)
		self.conn.commit()



	def __del__(self):
		self.conn.close()


