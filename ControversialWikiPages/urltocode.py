file = open("Urls.txt","r")
line = file.readlines()
List=[]
a=0
for i in line:
	b=i[30:]
	if (i!="Excptnthrn" and b[0:9]!="User_talk" and b[0:14]!="Wikipedia_talk" and b[0:4]!="Talk"):
		a=a+1
		temp=i[30:]
		List.append(temp)
		print(temp, end="")