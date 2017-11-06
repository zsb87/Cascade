i = 0
n = 0
j = 0
t = 3
while(i<10):
	n = n+1
	print('n:',n)
	while(j<5):
		print('j:',j)
		if t>0:
			t-=2
			print('t:',t)
		else:
			break

print('jump out of outest while loop!')
