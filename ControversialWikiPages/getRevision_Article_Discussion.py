import urllib2
import re
import socket
from itertools import izip

def check_ip(ip_addr, version): 
    if version == 4:
        try:
            socket.inet_aton(ip_addr)
        except socket.error:
            return 1
        else:
            return 0
 
    elif version == 6:
        try:
            socket.inet_pton(socket.AF_INET6, ip_addr)
        except socket.error:
            return 1
        else:
            return 0
def GetRevisions(pageTitle):
    url = "https://en.wikipedia.org/w/api.php?action=query&format=xml&prop=revisions&rvlimit=500&titles=" + pageTitle
    #url="https://en.wikipedia.org/wiki/Wikipedia:Sandbox"
    revisions = []                                        #list of all accumulated revisions
    next = ''                                             #information for the next request
    while True:
        response = urllib2.urlopen(url + next).read()     #web request
        revisions += re.findall('<rev [^>]*>', response)  #adds all revisions from the current request to the list

        cont = re.search('<continue rvcontinue="([^"]+)"', response)
        if not cont:                                      #break the loop if 'continue' element missing
            break

        next = "&rvcontinue=" + cont.group(1)             #gets the revision Id from which to start the next request
    
    usr_lst=[]
    l_edit=0
    print("Total "+str(len(revisions)))
    for i in revisions:
        arr=[]
        if(i.find('(indefinite)')):
            l_edit=1;
        arr=i.split();

        for j in range(0,len(arr)):        
            tmpl=[]
            tmpl=arr[j].split('=') 
            if len(tmpl)>1:      
                if tmpl[0]=='user':
                    #print(arr[j])
                    usr_lst.append(tmpl[1])           
        #print(i)
    usr_ls=[]
    count_an=0
    for i in usr_lst:
        usr_ls.append(i.split('"')[1])
    for i in usr_ls:
        if i=='annonymous':
            count_an=count_an+1  
        if i=='Annonymous':
            count_an=count_an+1           
        a=i.count(':')
        if a==7:
            if check_ip(i,6)==0:
                count_an=count_an+1
            
        b=i.count('.')
        if b==3:
            if check_ip(i,4)==0:
                count_an=count_an+1        
        #print(i)

    print("annonymous "+str(count_an))
    print("righteous "+str(len(set(usr_lst))))
    print("edit_size "+ str(l_edit))
    return revisions;
file = open("Urls.txt","r")
line = file.readlines()
List=[]
# for i in line:
#     if (i!="Excptnthrn"):
#         print(i.rstrip('\n'))
#         GetRevisions(i.rstrip('\n'))
        
with open("Urls.txt","r") as m, open("Urls2.txt","r") as d:
    for x, y in izip(m,d):
        x = x.strip()
        y = y.strip()
        print(x)
        GetRevisions(x.rstrip('\n'))
        GetRevisions(y.rstrip('\n'))
        print('')
        
