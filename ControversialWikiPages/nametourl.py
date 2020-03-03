import wikipedia

file = open("Names.txt", "r")
line = file.readlines()
List=[]
for i in line:
	temp=i.split("\t")
	List.append(temp[1])

for x in range(len(List)):
	try:
		page=wikipedia.page(List[x])
		link=page.url
		print(link)
	except Exception:
		print("Excptnthrn")

	

