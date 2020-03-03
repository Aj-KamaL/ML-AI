file = open("temp","r")
line = file.readlines()
List=[]
for i in line:
    if (i!="Excptnthrn"):
    	if i != '\n':
    		# print("1")
    		if "edit_size" not in i: 
    			print("3")
    			List.append(i);

    		elif "edit_size" in i:
    			List.append(i);
    			# print("2")
    			if len(List)>0:   
    				# print("1") 				
	    			for j in List:
	    				g=j.split()
	    				print(g)
	    				if len(g)>1:
	    					print (g[1],),
	    				else :
	    					print(g[0],),
    			print("")
    			List=[]


        