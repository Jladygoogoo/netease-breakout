import pymysql

from connect_db import MyConn


def upload_results():
	conn = MyConn()
	data = {}

	data["model_type"] = input("[Model Type]: ")

	params = []
	print("[Parameters Settings]")
	flag = 1
	while 1:
		param = input("[{}] param:".format(flag))
		if param=='': break
		value = input("[{}] value:".format(flag))
		params.append(": ".join((param, value)))
		flag += 1
	data["params"] = "; ".join(params)

	data["data_source"] = input("[Data Source]: ")

	evals = []
	print("[Evaluations]")
	flag = 1
	while 1:
		key = input("[{}] indicator:".format(flag))
		if key=='': break
		value = input("[{}] value:".format(flag))
		evals.append(": ".join((key, value)))
		flag += 1
	data["evaluations"] = "; ".join(evals)

	model_saved = int(input("[Model Saved (0/1)]: "))
	if model_saved == 1:
		data["model_saved"] = 1
		data["model_path"] = input("[Model Path]: ")
	else:
		data["model_saved"] = 0
		data["model_path"] = ""

	conn.insert(settings=data, table="results")


if __name__ == '__main__':
	upload_results()
