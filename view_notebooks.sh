# clear away current files
echo "[Clearing existing files]"
rm -r ViewDocs/*

# convert current files
echo "[Creating HTML files]"
jupyter nbconvert 01-Principles/*.ipynb
jupyter nbconvert 02-Simulation/*.ipynb
jupyter nbconvert 03-Data/*.ipynb
jupyter nbconvert 04-Visualization/*.ipynb
jupyter nbconvert 05-Learning/*.ipynb

tme=$(date +%Y%m%d)
cd 01-Principles/
for file in *.html; do mv "$file" "${file%.html}"_$tme.html; done
cd ../02-Simulation/
for file in *.html; do mv "$file" "${file%.html}"_$tme.html; done
cd ../03-Data/
for file in *.html; do mv "$file" "${file%.html}"_$tme.html; done
cd ../04-Visualization/
for file in *.html; do mv "$file" "${file%.html}"_$tme.html; done
cd ../05-Learning/
for file in *.html; do mv "$file" "${file%.html}"_$tme.html; done
cd ..

# move into viewdocs
mv 01-Principles/*.html ViewDocs/
mv 02-Simulation/*.html ViewDocs/
mv 03-Data/*.html ViewDocs/
mv 04-Visualization/*.html ViewDocs/
mv 05-Learning/*.html ViewDocs/