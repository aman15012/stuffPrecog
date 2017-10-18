import facebook

token = "--fill--"
graph = facebook.GraphAPI(access_token=token, version="2.9")

X = graph.get_connections(id='me', connection_name='feed', fields='with_tags,created_time', limit='1000')

friend_dict ={}

for i in range(len(X["data"])):
	try:
		A = X["data"][i]
	except:
		print("-Skipped-")
	if("with_tags" in A.keys()):
		if(A["created_time"][:4] in friend_dict):
			for j in A["with_tags"]["data"]:
				friend_dict[A["created_time"][:4]].add(j["name"])
		else:
			temp = {"me"}
			for j in A["with_tags"]["data"]:
				temp.add(j["name"])
			friend_dict[A["created_time"][:4]] = temp
			temp.remove("me")

for i in reversed(sorted(friend_dict.keys())):
	print(i,friend_dict[i])