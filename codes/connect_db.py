import pymysql

class MyConn:
	def __init__(self):
		self.conn = pymysql.connect(host="127.0.0.1", port=3306, user="root",
							db="NetEase_proxied", password="SFpqwnj285798,.")
		self.cache_count = 0

	def query(self, targets=None, conditions=None, sql=None, table="tracks"):
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

		return cursor.fetchall()

	def update(self, settings=None, conditions=None, sql=None, table="tracks"):
		if not sql:
			sql = "UPDATE {} SET".format(table)
			for i, k in enumerate(settings):
				sql += " {}=%s".format(k)
				if i<len(conditions)-1:
					sql += ","
			if conditions:
				sql += " WHERE"
				for i, k in enumerate(conditions):
					sql += " {}=%s".format(k)
					if i<len(conditions)-1:
						sql += " AND"

		with self.conn.cursor() as cursor:
			cursor.execute(sql, tuple(settings.values())+tuple(conditions.values()))

		self.conn.commit()

		# self.cache_count += 1
		# if self.cache_count == 100:
		# 	self.conn.commit()
		# 	self.cache_count = 0


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


