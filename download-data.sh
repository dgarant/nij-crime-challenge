mkdir -p data
cd data
wget -4 https://www.nij.gov/documents/crime-forecasting-challenge/portland-police-districts.zip
wget -4 https://www.nij.gov/Documents/crime-forecasting-challenge/110116_113016_Data.zip
wget -4 https://www.nij.gov/Documents/crime-forecasting-challenge/100116_103116_Data.zip
wget -4 https://www.nij.gov/Documents/crime-forecasting-challenge/090116_093016_Data.zip
wget -4 https://www.nij.gov/Documents/crime-forecasting-challenge/080116_083116_Data.zip
wget -4 https://www.nij.gov/documents/crime-forecasting-challenge/010116_073116_Data.zip
wget -4 https://www.nij.gov/documents/crime-forecasting-challenge/010115_123115_Data.zip
wget -4 https://www.nij.gov/documents/crime-forecasting-challenge/010114_123114_Data.zip
wget -4 https://www.nij.gov/documents/crime-forecasting-challenge/010113_123113_Data.zip
wget -4 https://www.nij.gov/documents/crime-forecasting-challenge/030112_123112_Data.zip
wget -4 https://www.nij.gov/Documents/crime-forecasting-challenge/EXAMPLE_SUBMISSION.zip
wget -4 https://calendar.travelportland.com/calendar.xml
wget -4 https://www.nij.gov/Documents/crime-forecasting-challenge/120116_123116_Data.zip
wget -4 https://www.nij.gov/Documents/crime-forecasting-challenge/010117_013117_Data.zip
wget -4 https://www.nij.gov/Documents/crime-forecasting-challenge/020117_021417_Data.zip
wget -4 https://www.nij.gov/Documents/crime-forecasting-challenge/021517_022117_Data.zip
wget -4 https://www.nij.gov/Documents/crime-forecasting-challenge/022217_022617_Data.zip

for f in *.zip; do
    unzip $f;
done


