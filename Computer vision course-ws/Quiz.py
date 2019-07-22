def calc_surface(lists,n):
     product=[]
     Area=[]
     for i in range(0,lists.__len__()-1):
        product.append((lists[i][0]['x']*lists[i+1][1]['y'])-(lists[i][1]['y']*lists[i+1][0]['x']))
     for j in product:
           Area.append(abs(j/2))
     Area.sort()
     Area.reverse()

     return Area[0:n]



A = [({'x':1}, {'y':2}), ({'x':5}, {'y':2}), ({'x':1},
{'y':3}), ({'x':1}, {'y':0})]
B = [({'x':2}, {'y':2}), ({'x':4}, {'y':3})]
C = [({'x':12}, {'y':5}), ({'x':3}, {'y':2}), ({'x':3},
{'y':7}), ({'x':2}, {'y':2})]

n = int(input('enter n:'))
print(calc_surface(C,n))

