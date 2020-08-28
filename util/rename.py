import glob
import os

for filename in glob.glob('*.jpg'):
    name, ext = os.path.splitext(filename)
    os.rename(filename, name + '_' + ext)

'''
dir=./Apple
class=Apple
cd $dir
i=0
for x in `ls *`
do
mv $x "$class"_"$i"_1.0_.jpg
let i+=1
done
'''
