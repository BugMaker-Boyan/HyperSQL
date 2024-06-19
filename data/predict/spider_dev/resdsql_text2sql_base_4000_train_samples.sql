select count ( * ) from singer
select count ( * ) from singer
select name , country , age from singer order by age desc
select name , country , age from singer order by age desc
select avg ( age ) , min ( age ) , max ( age ) from singer where country = 'France'
select avg ( age ) , min ( age ) , max ( age ) from singer where country = 'France'
select song_name , song_release_year from singer order by age asc limit 1
select song_name , song_release_year from singer order by age asc limit 1
select distinct country from singer where age > 20
select distinct country from singer where age > 20
select country , count ( * ) from singer group by country
select country , count ( * ) from singer group by country
select song_name from singer where age > ( select avg ( age ) from singer )
select song_name from singer where age > ( select avg ( age ) from singer )
select location , name from stadium where capacity between 5000 and 10000
select location , name from stadium where capacity between 5000 and 10000
select max ( capacity ) , avg ( average ) from stadium
select avg ( capacity ) , max ( capacity ) from stadium
select name , capacity from stadium group by average order by avg ( average ) desc limit 1
select name , capacity from stadium group by average order by avg ( average ) desc limit 1
select count ( * ) from concert where year = 2014 or year = 2015
select count ( * ) from concert where year = 2014 or year = 2015
select stadium.name , count ( * ) from stadium join concert on stadium.stadium_id = concert.stadium_id group by stadium.stadium_id
select stadium.stadium_id , count ( * ) from stadium join concert on stadium.stadium_id = concert.stadium_id group by stadium.stadium_id
select stadium.name , stadium.capacity from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year >= 2014 group by stadium.stadium_id order by count ( * ) desc limit 1
select stadium.name , stadium.capacity from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year > 2013 group by stadium.stadium_id order by count ( * ) desc limit 1
select year from concert group by year order by count ( * ) desc limit 1
select year from concert group by year order by count ( * ) desc limit 1
select name from stadium where stadium_id not in ( select stadium_id from concert )
select name from stadium where stadium_id not in ( select stadium_id from concert )
select country from singer where age > 40 intersect select country from singer where age < 30
select name from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year = 2014 except select stadium.name from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year = 2014
select name from stadium where stadium_id not in ( select stadium_id from concert where year = 2014 )
select concert.concert_name , concert.theme , count ( * ) from concert join singer_in_concert on concert.concert_id = singer_in_concert.concert_id group by concert.concert_name
select concert.concert_name , concert.theme , count ( * ) from concert join singer_in_concert on concert.concert_id = singer_in_concert.concert_id group by concert.concert_id
select singer.name , count ( * ) from singer_in_concert join singer on singer_in_concert.singer_id = singer.singer_id group by singer.singer_id
select singer.name , count ( * ) from singer_in_concert join singer on singer_in_concert.singer_id = singer.singer_id group by singer.singer_id
select singer.name from concert join singer_in_concert on concert.concert_id = singer_in_concert.concert_id join singer on singer_in_concert.singer_id = singer.singer_id where concert.year = 2014
select singer.name from singer join singer_in_concert on singer.singer_id = singer_in_concert.singer_id join concert on singer_in_concert.concert_id = concert.concert_id where concert.year = 2014
select name , country from singer where song_name like '%Hey%'
select name , country from singer where song_name like '%Hey%'
select stadium.name , stadium.location from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year = 2014 intersect select stadium.name , stadium.location from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year = 2015
select stadium.name , stadium.location from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year = 2014 intersect select stadium.name , stadium.location from stadium join concert on stadium.stadium_id = concert.stadium_id where concert.year = 2015
select count ( * ) from stadium join concert on stadium.stadium_id = concert.stadium_id where stadium.capacity = ( select max ( capacity ) from stadium )
select count ( * ) from stadium join concert on stadium.stadium_id = concert.stadium_id where stadium.capacity = ( select max ( capacity ) from stadium )
select count ( * ) from pets where weight > 10
select count ( * ) from pets where weight > 10
select weight from pets order by pet_age asc limit 1
select weight from pets order by pet_age asc limit 1
select max ( weight ) , pettype from pets group by pettype
select max ( weight ) , pettype from pets group by pettype
select count ( * ) from has_pet join pets on has_pet.petid = pets.petid join student on student.stuid = has_pet.stuid where student.age > 20
select count ( * ) from has_pet join pets on has_pet.petid = pets.petid join student on student.stuid = has_pet.stuid where student.age > 20
select count ( * ) from pets join has_pet on pets.petid = has_pet.petid join student on student.stuid = has_pet.stuid where student.sex = 'F'
select count ( * ) from has_pet join pets on has_pet.petid = pets.petid join student on student.stuid = has_pet.stuid where student.sex = 'F'
select count ( distinct pettype ) from pets
select count ( distinct pettype ) from pets
select student.fname from has_pet join student on has_pet.stuid = student.stuid join pets on has_pet.petid = pets.petid where pets.pettype = 'cat' or pets.pettype = 'dog'
select student.fname from has_pet join student on has_pet.stuid = student.stuid join pets on has_pet.petid = pets.petid where pets.pettype = 'cat' or pets.pettype = 'dog'
select student.fname from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where pets.pettype = 'cat' intersect select student.fname from student join has_pet on student.stuid = has_pet.stuid join pets on has_pet.petid = pets.petid where pets.pettype = 'dog'
sql placeholder
select major , age from student where stuid not in ( select stuid from has_pet join pets on has_pet.petid = pets.petid where pets.pettype = 'cat' )
select major , age from student where stuid not in ( select stuid from has_pet where petid = 'cat' )
select stuid from student except select stuid from has_pet join pets on has_pet.petid = pets.petid where pets.pettype = 'cat'
sql placeholder
select student.fname , student.age from student join has_pet on student.stuid = has_pet.stuid join pets on pets.petid = has_pet.petid where pets.pettype = 'dog' except select student.fname , student.age from student join has_pet on student.stuid = has_pet.stuid join pets on pets.petid = has_pet.petid where pets.pettype = 'cat'
select fname from student where stuid = 'dog' except select stuid from has_pet join pets on has_pet.petid = pets.petid where pets.pettype = 'cat'
select pettype , weight from pets order by pet_age asc limit 1
select pettype , weight from pets order by pet_age asc limit 1
select petid , weight from pets where pet_age > 1
select petid , weight from pets where pet_age > 1
select avg ( pet_age ) , max ( pet_age ) , pettype from pets group by pettype
select avg ( pet_age ) , max ( pet_age ) , pettype from pets group by pettype
select pettype , avg ( weight ) from pets group by pettype
select pettype , avg ( weight ) from pets group by pettype
select student.fname , student.age from has_pet join student on has_pet.stuid = student.stuid join pets on has_pet.petid = pets.petid
select distinct student.fname , student.age from student join has_pet on student.stuid = has_pet.stuid join pets on pets.petid = has_pet.petid where has_pet.petid = 1
select pets.petid from has_pet join pets on has_pet.petid = pets.petid join student on student.stuid = has_pet.stuid where student.lname = 'Smith'
select pets.petid from has_pet join pets on has_pet.petid = pets.petid join student on student.stuid = has_pet.stuid where student.lname = 'Smith'
select count ( * ) , stuid from has_pet group by stuid
select count ( * ) , stuid from student group by stuid
select student.fname , student.sex from student join has_pet on student.stuid = has_pet.stuid join pets on pets.petid = has_pet.petid group by has_pet.petid having count ( * ) > 1
select student.fname , student.sex from student join has_pet on student.stuid = has_pet.stuid join pets on pets.petid = has_pet.petid group by has_pet.petid having count ( * ) > 1
select lname from student where age = 3
select student.lname from has_pet join student on has_pet.stuid = student.stuid join pets on has_pet.petid = pets.petid where pets.pet_age = 3
select avg ( age ) from student where stuid not in ( select stuid from has_pet )
select avg ( age ) from student where stuid not in ( select stuid from has_pet )
select count ( * ) from continents
select count ( * ) from continents
select continents.contid , continents.continent , count ( * ) from continents join countries on continents.contid = countries.continent group by continents.contid
select continents.contid , countries.countryname , count ( * ) from continents join countries on continents.contid = countries.continent group by continents.contid
select count ( * ) from countries
select count ( * ) from countries
select car_makers.fullname , car_makers.id , count ( * ) from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id
select car_makers.fullname , car_makers.id , count ( * ) from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id
select model_list.model from cars_data join model_list on cars_data.id = model_list.model order by cars_data.horsepower asc limit 1
select model_list.model from cars_data join model_list on cars_data.id = model_list.model order by cars_data.horsepower asc limit 1
select model_list.model from cars_data join model_list on cars_data.id = model_list.model where cars_data.weight < ( select avg ( weight ) from cars_data )
select model_list.model from cars_data join model_list on cars_data.id = model_list.model where cars_data.weight < ( select avg ( weight ) from cars_data )
select car_makers.maker from car_makers join cars_data on car_makers.id = cars_data.id where cars_data.year = 1970
select distinct car_makers.maker from car_makers join cars_data on car_makers.id = cars_data.id where cars_data.year = 1970
sql placeholder
select car_makers.maker , cars_data.year from car_makers join cars_data on car_makers.id = cars_data.id order by cars_data.year asc limit 1
select distinct car_names.model from car_names join cars_data on car_names.makeid = cars_data.id where cars_data.year > 1980
select distinct model_list.model from model_list join cars_data on model_list.model = cars_data.id where cars_data.year > 1980
select continent , count ( * ) from continents group by continent
select continents.continent , count ( * ) from continents join car_makers on continents.contid = car_makers.country group by continents.continent
select countries.countryname from car_makers join countries on car_makers.country = countries.countryid group by car_makers.country order by count ( * ) desc limit 1
select countries.countryname from car_makers join countries on car_makers.country = countries.countryid group by car_makers.country order by count ( * ) desc limit 1
select count ( * ) , maker from car_makers group by maker
select count ( * ) , car_makers.id , car_makers.fullname from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id
sql placeholder
select sum ( cars_data.accelerate ) from cars_data join car_names on cars_data.id = car_names.makeid where car_names.make = 'amc hornet' and car_names.make = 'sw'
select count ( * ) from countries join car_makers on countries.countryid = car_makers.country where countries.countryname = 'france'
select count ( * ) from countries join car_makers on countries.countryid = car_makers.country where countries.countryname = 'france'
select count ( * ) from countries join car_makers on countries.countryid = car_makers.country where countries.countryname = 'usa'
select count ( * ) from car_makers where country = 'USA'
select avg ( mpg ) from cars_data where cylinders = 4
select avg ( mpg ) from cars_data where cylinders = 4
select min ( weight ) from cars_data where year = 1974 and cylinders = 8
select min ( weight ) from cars_data where year = 1974
select car_makers.maker , model_list.model from model_list join car_makers on model_list.maker = car_makers.id
select car_makers.maker , model_list.model from model_list join car_makers on model_list.maker = car_makers.id
select countries.countryname , car_makers.id from car_makers join countries on car_makers.country = countries.countryid group by car_makers.country having count ( * ) >= 1
select countries.countryname , car_makers.id from countries join car_makers on countries.countryid = car_makers.country group by car_makers.id having count ( * ) >= 1
select count ( * ) from cars_data where horsepower > 150
select count ( * ) from cars_data where horsepower > 150
select avg ( weight ) , year from cars_data group by year
select avg ( weight ) , avg ( year ) from cars_data group by year
select countries.countryname from countries join continents on countries.continent = continents.contid join car_makers on countries.countryid = car_makers.country where continents.continent = 'europe' group by countries.countryname having count ( * ) >= 3
select countries.countryname from countries join continents on countries.continent = continents.contid join car_makers on car_makers.maker = car_makers.id where continents.continent = 'europe' group by countries.countryname having count ( * ) >= 3
select max ( cars_data.horsepower ) , car_names.make from cars_data join car_names on cars_data.id = car_names.makeid where cars_data.cylinders = 3
sql placeholder
select model_list.model from model_list join cars_data on model_list.model = cars_data.id order by cars_data.mpg desc limit 1
select model_list.model from cars_data join model_list on cars_data.id = model_list.model order by cars_data.mpg desc limit 1
select avg ( horsepower ) from cars_data where year < 1980
select avg ( horsepower ) from cars_data where year < 1980
sql placeholder
sql placeholder
select max ( accelerate ) , cylinders from cars_data group by cylinders
select max ( accelerate ) , cylinders from cars_data group by cylinders
select model_list.model from car_names join model_list on car_names.model = model_list.model group by model_list.model order by count ( * ) desc limit 1
select model from model_list group by model order by count ( * ) desc limit 1
select count ( * ) from cars_data where cylinders > 4
select count ( * ) from cars_data where cylinders > 4
select count ( * ) from cars_data where year = 1980
select count ( * ) from cars_data where year = 1980
select count ( * ) from car_makers where fullname = 'American Motor Company'
select count ( * ) from car_makers join model_list on car_makers.id = model_list.maker where car_makers.fullname = 'American Motor Company'
select car_makers.fullname , car_makers.id from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id having count ( * ) > 3
select car_makers.maker , car_makers.id from car_makers join model_list on car_makers.id = model_list.maker group by model_list.maker having count ( * ) > 3
sql placeholder
sql placeholder
select year from cars_data where weight < 3000 intersect select year from cars_data where weight > 4000
select distinct year from cars_data where weight < 4000 intersect select distinct year from cars_data where weight > 3000
select horsepower from cars_data order by accelerate desc limit 1
select horsepower from cars_data order by accelerate desc limit 1
select cylinders from cars_data where accelerate = 'Min ( accelerate )' order by cylinders asc limit 1
select cylinders from cars_data where accelerate = 'Voyager' order by accelerate asc limit 1
select count ( * ) from cars_data where accelerate > ( select max ( accelerate ) from cars_data )
select count ( * ) from cars_data where accelerate > ( select max ( accelerate ) from cars_data )
select count ( * ) from car_makers group by country having count ( * ) > 2
select count ( * ) from car_makers group by country having count ( * ) > 2
select count ( * ) from cars_data where cylinders > 6
select count ( * ) from cars_data where cylinders > 6
select model_list.model from cars_data join model_list on cars_data.id = model_list.model where cars_data.cylinders = 4 order by cars_data.horsepower desc limit 1
select cylinders from cars_data where horsepower = 4 order by horsepower desc limit 1
sql placeholder
select car_names.makeid , car_names.model from cars_data join car_names on cars_data.id = car_names.makeid where cars_data.horsepower < 4
select max ( mpg ) from cars_data where cylinders >= 8 or year < 1980
select max ( mpg ) from cars_data where cylinders = 8 or year < 1980
select model_list.model from model_list join car_makers on model_list.maker = car_makers.id where car_makers.fullname = 'Ford Motor Company' except select model_list.model from model_list join car_makers on model_list.maker = car_makers.id where car_makers.fullname = 'Ford Motor Company'
select distinct model_list.model from model_list join car_makers on model_list.maker = car_makers.id where car_makers.fullname = 'Ford Motor Company' except select distinct model_list.model from model_list join car_makers on model_list.maker = car_makers.id where car_makers.fullname = 'Ford Motor Company'
select countryname from countries where countryid not in ( select country from car_makers )
select countryname from countries where countryid not in ( select country from car_makers )
select car_makers.id , car_makers.maker from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id having count ( * ) >= 2 union select car_makers.id , car_makers.maker from car_makers join model_list on car_makers.id = model_list.model group by car_makers.id having count ( * ) >= 3
select car_makers.id , car_makers.maker from car_makers join model_list on car_makers.id = model_list.maker group by car_makers.id having count ( * ) >= 2 union select car_makers.id , car_makers.maker from car_makers join model_list on car_makers.id = model_list.model group by car_makers.id having count ( * ) > 3
select countries.countryid , countries.countryname from countries join car_makers on countries.countryid = car_makers.country where car_makers.maker = 'fiat' group by countries.countryid having count ( * ) > 3 union select countries.countryid , countries.countryname from countries join model_list on countries.countryid = model_list.maker where model_list.model = 'fiat'
sql placeholder
select country from airlines where airline = 'JetBlue Airways'
select country from airlines where abbreviation = 'JetBlue Airways'
select abbreviation from airlines where airline = 'JetBlue Airways'
select abbreviation from airlines where abbreviation = 'JetBlue Airways'
select airline , abbreviation from airlines where country = 'USA'
select airline , abbreviation from airlines where country = 'USA'
select airportcode , airportname from airports where city = 'Anthony'
select airportcode , airportname from airports where city = 'Anthony'
select count ( * ) from airlines
select count ( * ) from airlines
select count ( * ) from airports
select count ( * ) from airports
select count ( * ) from flights
select count ( * ) from flights
select airline from airlines where abbreviation = 'UAL'
select airline from airlines where abbreviation = 'UAL'
select count ( * ) from airlines where country = 'USA'
select count ( * ) from airlines where country = 'USA'
select city , country from airports where airportname = 'Alton'
select city , country from airports where airportname = 'Alton'
select airportname from airports where airportcode = 'AKO'
select airportname from airports where airportcode = 'AKO'
select airportname from airports where city = 'Aberdeen'
select airportname from airports where city = 'Aberdeen'
select count ( * ) from flights where sourceairport = 'APG'
select count ( * ) from flights where sourceairport = 'APG'
select count ( * ) from flights where destairport = 'ATO'
select count ( * ) from flights where sourceairport = 'ATO'
select count ( * ) from airports join flights on airports.airportcode = flights.sourceairport where airports.city = 'Aberdeen'
select count ( * ) from flights where sourceairport = 'Aberdeen'
select count ( * ) from airports where city = 'Aberdeen'
select count ( * ) from airports join flights on airports.airportcode = flights.destairport where airports.city = 'Aberdeen'
select count ( * ) from airports join flights on airports.airportcode = flights.destairport where airports.city = 'Aberdeen' and airports.airportname = 'Ashley'
select count ( * ) from airports join flights on airports.airportcode = flights.destairport where airports.city = 'Aberdeen' and airports.airportname = 'Ashley'
select count ( * ) from airlines join flights on airlines.airline = flights.airline where airlines.airline = 'JetBlue Airways'
select count ( * ) from airlines where airline = 'JetBlue Airways'
select count ( * ) from airlines join flights on airlines.airline = flights.airline where flights.destairport = 'ASY' and airlines.airline = 'United Airlines'
select count ( * ) from airlines join flights on airlines.airline = flights.airline where flights.destairport = 'ASY'
select count ( * ) from airlines join flights on airlines.airline = flights.airline where flights.sourceairport = 'AHD'
select count ( * ) from airlines join flights on airlines.airline = flights.airline where flights.sourceairport = 'AHD'
sql placeholder
select count ( * ) from flights join airports on flights.destairport = airports.airportcode join airlines on flights.airline = airlines.airline where airports.city = 'Aberdeen' and airlines.airline = 'United Airlines'
select city from airports group by city order by count ( * ) desc limit 1
select city from airports group by city order by count ( * ) desc limit 1
select city from airports group by city order by count ( * ) desc limit 1
select city from airports group by city order by count ( * ) desc limit 1
select airports.airportcode from airports join flights on airports.airportcode = flights.sourceairport group by flights.sourceairport order by count ( * ) desc limit 1
select airports.airportcode from airports join flights on airports.airportcode = flights.sourceairport group by flights.sourceairport order by count ( * ) desc limit 1
select airports.airportcode from airports join flights on airports.airportcode = flights.sourceairport group by flights.sourceairport order by count ( * ) asc limit 1
select airports.airportcode from airports join flights on airports.airportcode = flights.sourceairport group by flights.sourceairport order by count ( * ) asc limit 1
select airlines.airline from flights join airlines on flights.airline = airlines.airline group by airlines.airline order by count ( * ) desc limit 1
select airlines.airline from airlines join flights on airlines.airline = flights.airline group by airlines.airline order by count ( * ) desc limit 1
select airlines.abbreviation , airlines.country from airlines join flights on airlines.airline = flights.airline group by flights.airline order by count ( * ) asc limit 1
select abbreviation from airlines group by abbreviation order by count ( * ) asc limit 1
select airlines.airline from airlines join flights on airlines.airline = flights.airline where flights.sourceairport = 'AHD'
select airlines.airline from flights join airlines on flights.airline = airlines.airline where flights.sourceairport = 'AHD'
select airlines.airline from airlines join flights on airlines.airline = flights.airline where flights.sourceairport = 'AHD'
select airlines.airline from flights join airlines on flights.airline = airlines.airline where flights.destairport = 'AHD' and flights.sourceairport = 'AHD'
select airlines.airline from airlines join flights on airlines.airline = flights.airline where flights.sourceairport = 'APG' intersect select airlines.airline from airlines join flights on airlines.airline = flights.airline where flights.sourceairport = 'CVO'
select airlines.airline from airlines join flights on airlines.airline = flights.airline where flights.sourceairport = 'APG' intersect select airlines.airline from airlines join flights on airlines.airline = flights.airline where flights.sourceairport = 'CVO'
sql placeholder
sql placeholder
select airlines.airline from airlines join flights on airlines.airline = flights.airline group by flights.airline having count ( * ) >= 10
select airlines.airline from airlines join flights on airlines.airline = flights.airline group by flights.airline having count ( * ) >= 10
select airlines.airline from airlines join flights on airlines.airline = flights.airline group by flights.airline having count ( * ) < 200
select airlines.airline from airlines join flights on airlines.airline = flights.airline group by flights.airline having count ( * ) < 200
select flights.flightno from airlines join flights on airlines.airline = flights.airline where airlines.airline = 'United Airlines'
select flights.flightno from airlines join flights on airlines.airline = flights.airline where airlines.airline = 'United Airlines'
select flightno from flights where sourceairport = 'APG'
select flightno from flights where sourceairport = 'APG'
select flightno from flights where destairport = 'APG'
select flightno from flights where sourceairport = 'APG'
select flights.flightno from airports join flights on airports.airportcode = flights.sourceairport where airports.city = 'Aberdeen'
select flightno from flights where sourceairport = 'Aberdeen'
select flights.flightno from airports join flights on airports.airportcode = flights.sourceairport where airports.city = 'Aberdeen'
select flightno from flights where destairport = 'Aberdeen'
select count ( * ) from airports where city = 'Aberdeen' or city = 'Abilene'
select count ( * ) from airports where city = 'Aberdeen' or city = 'Abilene'
select airportname from airports where airportcode not in ( select airportcode from flights ) or airports.airportcode not in ( select airportcode from flights )
select airportname from airports except select airports.airportname from airports join flights on airports.airportcode = flights.sourceairport where flights.flightno = 'Departing' or flights.airline = 'Atlanta'
select count ( * ) from employee
select count ( * ) from employee
select name from employee order by age asc
select name from employee order by age asc
select count ( * ) , city from employee group by city
select count ( * ) , city from employee group by city
select city from employee where age < 30 group by city having count ( * ) > 1
select city from employee where age < 30 group by city having count ( * ) > 1
select count ( * ) , location from shop group by location
select location , count ( * ) from shop group by location
select manager_name , district from shop order by number_products desc limit 1
select manager_name , district from shop order by number_products desc limit 1
select min ( number_products ) , max ( number_products ) from shop
select min ( number_products ) , max ( number_products ) from shop
select name , location , district from shop order by number_products desc
select name , location , district from shop order by number_products desc
select name from shop where number_products > ( select avg ( number_products ) from shop )
select name from shop where number_products > ( select avg ( number_products ) from shop )
select employee.name from evaluation join employee on evaluation.employee_id = employee.employee_id group by evaluation.employee_id order by count ( * ) desc limit 1
select employee.name from employee join evaluation on employee.employee_id = evaluation.employee_id group by evaluation.employee_id order by count ( * ) desc limit 1
select employee.name from employee join evaluation on employee.employee_id = evaluation.employee_id order by evaluation.bonus desc limit 1
select employee.name from employee join evaluation on employee.employee_id = evaluation.employee_id order by evaluation.bonus desc limit 1
select name from employee where employee_id not in ( select employee_id from evaluation )
select name from employee where employee_id not in ( select employee_id from evaluation )
select shop.name from hiring join shop on hiring.shop_id = shop.shop_id group by hiring.shop_id order by count ( * ) desc limit 1
select shop.name from employee join hiring on employee.employee_id = hiring.shop_id join shop on hiring.shop_id = shop.shop_id group by hiring.shop_id order by count ( * ) desc limit 1
select name from shop where shop_id not in ( select shop_id from hiring )
select name from shop where shop_id not in ( select shop_id from employee )
select count ( * ) , shop.name from shop join hiring on shop.shop_id = hiring.shop_id group by hiring.shop_id
select count ( * ) , shop.name from shop join hiring on shop.shop_id = hiring.shop_id group by hiring.shop_id
select sum ( bonus ) from evaluation
select sum ( bonus ) from evaluation
select * from hiring
select * from hiring
select district from shop where number_products < 3000 intersect select district from shop where number_products > 10000
select district from shop where number_products < 3000 intersect select district from shop where number_products > 10000
select count ( distinct location ) from shop
select count ( distinct location ) from shop
select count ( * ) from documents
select count ( * ) from documents
select document_id , document_name , document_description from documents
select document_id , document_name , document_description from documents
select document_name , template_id from documents where document_description like '%w%'
select document_name , template_id from documents where document_description like '%w%'
select document_id , template_id , document_description from documents where document_name = 'Robbin CV'
select document_id , template_id , document_description from documents where document_name = 'Robbin CV'
select count ( distinct template_id ) from documents
select count ( distinct template_id ) from documents
select count ( * ) from documents join templates on documents.template_id = templates.template_id where templates.template_type_code = 'PPT'
select count ( * ) from documents join templates on documents.template_id = templates.template_id where templates.template_type_code = 'PPT'
select template_id , count ( * ) from documents group by template_id
select template_id , count ( * ) from documents group by template_id
select templates.template_id , templates.template_type_code from templates join documents on templates.template_id = documents.template_id group by templates.template_id order by count ( * ) desc limit 1
select templates.template_id , templates.template_type_code from templates join documents on templates.template_id = documents.template_id group by templates.template_id order by count ( * ) desc limit 1
select template_id from documents group by template_id having count ( * ) > 1
select template_id from documents group by template_id having count ( * ) > 1
select template_id from templates except select template_id from documents
select template_id from templates except select template_id from documents
select count ( * ) from templates
select count ( * ) from templates
select template_id , version_number , template_type_code from templates
select template_id , version_number , template_type_code from templates
select distinct template_type_code from templates
select distinct template_type_code from ref_template_types
select template_id from templates where template_type_code = 'PP' or template_type_code = 'PPT'
select template_id from templates where template_type_code = 'PP' or template_type_code = 'PPT'
select count ( * ) from templates where template_type_code = 'CV'
select count ( * ) from templates where template_type_code = 'CV'
select version_number , template_type_code from templates where version_number > 5
select version_number , template_type_code from templates where version_number > 5
select template_type_code , count ( * ) from templates group by template_type_code
select template_type_code , count ( * ) from templates group by template_type_code
select template_type_code from templates group by template_type_code order by count ( * ) desc limit 1
select template_type_code from templates group by template_type_code order by count ( * ) desc limit 1
select template_type_code from templates group by template_type_code having count ( * ) < 3
select template_type_code from templates group by template_type_code having count ( * ) < 3
select version_number , template_type_code from templates order by version_number asc limit 1
select min ( version_number ) , template_type_code from templates group by version_number
select templates.template_type_code from templates join documents on templates.template_id = documents.template_id where documents.document_name = 'Data base'
select templates.template_type_code from templates join documents on templates.template_id = documents.template_id where documents.document_name = 'Data base'
select documents.document_name from documents join templates on documents.template_id = templates.template_id where templates.template_type_code = 'BK'
select documents.document_name from documents join templates on documents.template_id = templates.template_id where templates.template_type_code = 'BK'
select template_type_code , count ( * ) from templates group by template_type_code
select template_type_code , count ( * ) from templates group by template_type_code
select templates.template_type_code from documents join templates on documents.template_id = templates.template_id group by templates.template_type_code order by count ( * ) desc limit 1
select template_type_code from templates group by template_type_code order by count ( * ) desc limit 1
select template_type_code from templates where template_id not in ( select template_id from documents )
select template_type_code from templates where template_id not in ( select template_id from documents )
select template_type_code , template_type_description from ref_template_types
select template_type_code , template_type_description from ref_template_types
select ref_template_types.template_type_description from ref_template_types join templates on ref_template_types.template_type_code = templates.template_type_code where templates.template_type_code = 'AD'
select ref_template_types.template_type_description from ref_template_types join templates on ref_template_types.template_type_code = templates.template_type_code where ref_template_types.template_type_code = 'AD'
select template_type_code from ref_template_types where template_type_description = 'Book'
select template_type_code from ref_template_types where template_type_description = 'Book'
select distinct ref_template_types.template_type_description from ref_template_types join templates on ref_template_types.template_type_code = templates.template_type_code where templates.date_effective_to in ( select distinct document_id from documents )
select distinct document_description from documents
select templates.template_id from ref_template_types join templates on templates.template_id = ref_template_types.template_type_code where ref_template_types.template_type_description = 'Presentation'
sql placeholder
select count ( * ) from paragraphs
select count ( * ) from paragraphs
select count ( * ) from documents join paragraphs on documents.document_id = paragraphs.document_id where documents.document_name = 'Summer Show'
select count ( * ) from documents join paragraphs on documents.document_id = paragraphs.document_id where documents.document_name = 'Summer Show'
select other_details from paragraphs where paragraph_text = 'Korea'
select other_details from paragraphs where paragraph_text = 'Korea'
select paragraphs.paragraph_id , paragraphs.paragraph_text from paragraphs join documents on paragraphs.document_id = documents.document_id where documents.document_name = 'Welcome to NY'
select paragraphs.paragraph_id , paragraphs.paragraph_text from paragraphs join documents on paragraphs.document_id = documents.document_id where documents.document_name = 'Welcome to NY'
select paragraphs.paragraph_text from paragraphs join documents on paragraphs.document_id = documents.document_id where documents.document_name = 'Customer reviews'
select paragraphs.paragraph_text from paragraphs join documents on paragraphs.document_id = documents.document_id where documents.document_name = 'Customer reviews'
select document_id , count ( * ) from paragraphs group by document_id order by count ( * ) asc
select document_id , count ( * ) from paragraphs group by document_id order by document_id asc
select document_id , document_name , count ( * ) from documents group by document_id
select document_id , document_name , count ( * ) from documents group by document_id
select document_id from paragraphs group by document_id having count ( * ) >= 2
select document_id from paragraphs group by document_id having count ( * ) >= 2
select documents.document_id , documents.document_name from documents join paragraphs on documents.document_id = paragraphs.document_id group by documents.document_id order by count ( * ) desc limit 1
select documents.document_id , documents.document_name from documents join paragraphs on documents.document_id = paragraphs.document_id group by documents.document_id order by count ( * ) desc limit 1
select document_id from paragraphs group by document_id order by count ( * ) asc limit 1
select document_id from paragraphs group by document_id order by count ( * ) asc limit 1
select document_id from paragraphs group by document_id having count ( * ) between 1 and 2
select document_id from paragraphs group by document_id having count ( * ) between 1 and 2
select document_id from paragraphs where paragraph_text = 'Brazil' intersect select document_id from paragraphs where paragraph_text = 'Ireland'
select document_id from paragraphs where paragraph_text = 'Brazil' intersect select document_id from paragraphs where paragraph_text = 'Ireland'
select count ( * ) from teacher
select count ( * ) from teacher
select name from teacher order by age asc
select name from teacher order by age asc
select age , hometown from teacher
select age , hometown from teacher
select name from teacher where hometown != 'Little Lever Urban District'
select name from teacher where hometown != ''
select name from teacher where age = 32 or age = 33
select name from teacher where age = 32 or age = 33
select hometown from teacher order by age asc limit 1
select hometown from teacher order by age asc limit 1
select hometown , count ( * ) from teacher group by hometown
select hometown , count ( * ) from teacher group by hometown
select hometown from teacher group by hometown order by count ( * ) desc limit 1
select hometown from teacher group by hometown order by count ( * ) desc limit 1
select hometown from teacher group by hometown having count ( * ) >= 2
select hometown from teacher group by hometown having count ( * ) >= 2
select teacher.name , course_arrange.course_id from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id join course on course_arrange.course_id = course.course_id
select teacher.name , course.course from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id join course on course_arrange.course_id = course.course_id
select teacher.name , course.course from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id join course on course_arrange.course_id = course.course_id order by teacher.name asc
select teacher.name , course.course from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id join course on course_arrange.course_id = course.course_id order by teacher.name asc
select teacher.name from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id join course on course_arrange.course_id = course.course_id where course.course = 'Math'
select teacher.name from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id join course on course_arrange.course_id = course.course_id where course.course = 'Math'
select teacher.name , count ( * ) from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id join course on course_arrange.course_id = course.course_id group by course_arrange.teacher_id
select teacher.name , count ( * ) from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id join course on course_arrange.course_id = course.course_id group by course_arrange.teacher_id
select teacher.name from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id join course on course_arrange.course_id = course.course_id group by course_arrange.teacher_id having count ( * ) >= 2
select teacher.name from course_arrange join teacher on course_arrange.teacher_id = teacher.teacher_id join course on course_arrange.course_id = course.course_id group by course_arrange.teacher_id having count ( * ) >= 2
select name from teacher where teacher_id not in ( select teacher_id from course_arrange )
select name from teacher where teacher_id not in ( select teacher_id from course_arrange )
select count ( * ) from visitor where age < 30
select name from visitor where level_of_membership > 4 order by level_of_membership desc
select avg ( age ) from visitor where level_of_membership != 4
select name , level_of_membership from visitor where level_of_membership > 4 order by age asc
select museum.museum_id , museum.name from museum join visit on museum.museum_id = visit.museum_id group by museum.museum_id order by count ( * ) desc limit 1
select avg ( num_of_staff ) from museum where open_year < 2009
select open_year , num_of_staff from museum where name = 'Plaza Museum'
select name from museum where num_of_staff > ( select min ( num_of_staff ) from museum where open_year > 2010 )
select visitor.id , visitor.name , visitor.age from visitor join visit on visitor.id = visit.visitor_id group by visit.visitor_id having count ( * ) > 1
select visitor.id , visitor.name , visitor.level_of_membership from visitor join visit on visitor.id = visit.visitor_id group by visit.visitor_id order by sum ( total_spent ) desc limit 1
select museum.museum_id , museum.name from museum join visit on museum.museum_id = visit.museum_id group by visit.museum_id order by count ( * ) desc limit 1
select name from museum where museum_id not in ( select museum_id from visit )
select visitor.name , visitor.age from visitor join visit on visitor.id = visit.visitor_id group by visit.visitor_id order by count ( * ) desc limit 1
select avg ( num_of_ticket ) , max ( num_of_ticket ) from visit
select sum ( total_spent ) from visit join visitor on visit.visitor_id = visitor.id where visitor.level_of_membership = 1
select visitor.name from visitor join visit on visitor.id = visit.visitor_id join museum on museum.museum_id = visit.museum_id where museum.open_year < 2009 intersect select visitor.name from visitor join visit on visitor.id = visit.visitor_id join museum on museum.museum_id = visit.museum_id where museum.open_year > 2011
select count ( * ) from visitor where id not in ( select visitor_id from visit join museum on visit.museum_id = museum.museum_id where museum.open_year > 2010 )
select count ( * ) from museum where open_year > 2013 or open_year < 2008
select count ( * ) from players
select count ( * ) from players
select count ( * ) from matches
select count ( * ) from matches
select first_name , birth_date from players where country_code = 'USA'
select first_name , birth_date from players where country_code = 'USA'
select avg ( loser_age ) , avg ( winner_age ) from matches
select avg ( loser_age ) , avg ( winner_age ) from matches
select avg ( winner_rank ) from matches
select avg ( winner_rank ) from matches
select min ( loser_rank ) from matches
select best_of from matches
select count ( distinct country_code ) from players
select count ( distinct country_code ) from players
select count ( distinct loser_name ) from matches
select count ( distinct loser_name ) from matches
select tourney_name from matches group by tourney_name having count ( * ) > 10
select tourney_name from matches group by tourney_name having count ( * ) > 10
select winner_name from matches where year = 2013 intersect select winner_name from matches where year = 2016
select players.first_name , players.last_name from players join matches on players.player_id = matches.winner_id where matches.year = 2013 intersect select players.first_name , players.last_name from players join matches on players.player_id = matches.winner_id where matches.year = 2016
select count ( * ) from matches where year = 2013 or year = 2016
select count ( * ) from matches where year = 2013 or year = 2016
select players.country_code , players.first_name from players join matches on players.player_id = matches.winner_id where matches.tourney_name = 'WTA Championships' intersect select players.country_code , players.first_name from players join matches on players.player_id = matches.winner_id where matches.tourney_name = 'Australian Open'
select players.first_name , players.country_code from players join matches on players.player_id = matches.winner_id where matches.tourney_name = 'Australian Open' intersect select players.first_name , players.country_code from players join matches on players.player_id = matches.winner_id where matches.tourney_name = 'WTA Championships'
sql placeholder
select first_name , country_code from players order by first_name desc limit 1
select first_name , last_name from players order by birth_date asc
select first_name , last_name from players order by birth_date asc
select first_name , last_name from players where hand = 'L' order by birth_date asc
select first_name , last_name from players where birth_date = 'Date' order by birth_date asc
select players.first_name , players.country_code from rankings join players on rankings.player_id = players.player_id group by rankings.player_id order by count ( * ) desc limit 1
select players.first_name , players.country_code from rankings join players on rankings.player_id = players.player_id group by rankings.player_id order by count ( * ) desc limit 1
select year from matches group by year order by count ( * ) desc limit 1
select year from matches group by year order by count ( * ) desc limit 1
select matches.winner_name , matches.winner_rank_points from matches join rankings on matches.winner_id = rankings.player_id group by matches.winner_id order by count ( * ) desc limit 1
select winner_name , count ( * ) from matches group by winner_name order by count ( * ) desc limit 1
select matches.winner_name from rankings join matches on rankings.player_id = matches.winner_id where matches.tourney_name = 'Australian Open' order by rankings.ranking_points desc limit 1
select matches.winner_name from rankings join matches on rankings.player_id = matches.winner_id where matches.tourney_name = 'Australian Open' group by matches.winner_name order by count ( * ) desc limit 1
select loser_name , winner_name from matches order by minutes desc limit 1
select winner_name , loser_name from matches order by minutes desc limit 1
select avg ( rankings.ranking ) , players.first_name from rankings join players on rankings.player_id = players.player_id group by players.player_id
select players.first_name , avg ( rankings.ranking ) from rankings join players on rankings.player_id = players.player_id group by players.first_name
select sum ( ranking_points ) , players.first_name from rankings join players on rankings.player_id = players.player_id group by players.player_id
select players.first_name , sum ( rankings.ranking_points ) from rankings join players on rankings.player_id = players.player_id group by rankings.player_id
select count ( * ) , country_code from players group by country_code
select country_code , count ( * ) from players group by country_code
select country_code from players group by country_code order by count ( * ) desc limit 1
select country_code from players group by country_code order by count ( * ) desc limit 1
select country_code from players where country_code > 50 group by country_code having count ( * ) > 50
select country_code from players where country_code > 50
select ranking_date , count ( * ) from rankings group by ranking_date
select ranking_date , sum ( tours ) from rankings group by ranking_date
select count ( * ) , year from matches group by year
select count ( * ) , year from matches group by year
select winner_name , winner_rank from matches order by winner_age asc limit 3
select winner_name , winner_rank from matches order by winner_age asc limit 3
select count ( distinct winner_hand ) from matches where tourney_name = 'WTA Championships' intersect select count ( distinct winner_hand ) from matches where tourney_name = 'Los Angeles'
select count ( * ) from matches join players on matches.winner_id = players.player_id where matches.tourney_name = 'WTA Championships' and players.hand = 'left'
select players.first_name , players.country_code , players.birth_date from players join matches on players.player_id = matches.winner_id join rankings on rankings.player_id = matches.winner_id order by rankings.ranking_points desc limit 1
select players.first_name , players.country_code , players.birth_date from players join matches on players.player_id = matches.winner_id join rankings on matches.winner_id = rankings.player_id group by matches.winner_id order by count ( * ) desc limit 1
select hand , count ( * ) from players group by hand
select hand , count ( * ) from players group by hand
select count ( * ) from ship where disposition_of_ship = 'Captured'
select name , tonnage from ship order by name desc
select name , date , result from battle
select max ( killed ) , min ( killed ) , caused_by_ship_id from death group by caused_by_ship_id
select injured , avg ( injured ) from death group by injured
select death.killed , death.injured from death join ship on death.caused_by_ship_id = ship.id where ship.tonnage = 't'
select name , result from battle where bulgarian_commander != 'Boril'
select distinct battle.id , battle.name from battle join ship on battle.id = ship.lost_in_battle where ship.ship_type = 'Brig'
select battle.id , battle.name from battle join death on battle.id = death.caused_by_ship_id group by battle.id having sum ( death.killed ) > 10
select ship.id , ship.name from ship join death on ship.id = death.caused_by_ship_id group by death.caused_by_ship_id order by sum ( death.injured ) desc limit 1
select distinct name from battle where bulgarian_commander = 'Kaloyan' and latin_commander = 'Baldwin I'
select count ( distinct result ) from battle
select count ( * ) from battle where id not in ( select lost_in_battle from ship where tonnage = '225' )
select battle.name , battle.date from battle join ship on battle.id = ship.lost_in_battle where ship.name = 'Lettice' intersect select battle.name , battle.date from battle join ship on battle.id = ship.id where ship.name = 'HMS Atalanta'
select name , result , bulgarian_commander from battle where id not in ( select lost_in_battle from ship where location = 'English Channel' )
select note from death where note like '%East%'
sql placeholder
select line_1 , line_2 from addresses
select count ( * ) from courses
select count ( * ) from courses
select course_description from courses where course_name = 'math'
select course_description from courses where course_name = 'math'
select zip_postcode from addresses where city = 'Port Chelsea'
select zip_postcode from addresses where city = 'Port Chelsea'
select departments.department_name , departments.department_id from departments join degree_programs on departments.department_id = degree_programs.department_id group by departments.department_id order by count ( * ) desc limit 1
select departments.department_name , departments.department_id from departments join degree_programs on departments.department_id = degree_programs.department_id group by departments.department_id order by count ( * ) desc limit 1
select count ( distinct department_id ) from degree_programs
select count ( distinct department_id ) from degree_programs
select count ( distinct degree_summary_name ) from degree_programs
select count ( distinct degree_program_id ) from degree_programs
select count ( * ) from degree_programs join departments on degree_programs.department_id = departments.department_id where departments.department_description = 'Engineering'
select count ( * ) from departments join degree_programs on departments.department_id = degree_programs.department_id where departments.department_name = 'Engineering'
select section_name , section_description from sections
select section_name , section_description from sections
select courses.course_name , courses.course_id from courses join sections on courses.course_id = sections.course_id group by courses.course_id having count ( * ) <= 2
select courses.course_name , courses.course_id from courses join sections on courses.course_id = sections.course_id group by courses.course_id having count ( * ) < 2
select section_name from sections order by section_name desc
select section_name from sections order by section_name desc
select semesters.semester_name , student_enrolment.student_id from semesters join student_enrolment on semesters.semester_id = student_enrolment.semester_id group by student_enrolment.student_id order by count ( * ) desc limit 1
select semesters.semester_name , semesters.semester_id from semesters join student_enrolment on semesters.semester_id = student_enrolment.semester_id group by semesters.semester_id order by count ( * ) desc limit 1
select department_description from departments where department_name like '%computer%'
select department_description from departments where department_name like '%computer%'
select student_enrolment.student_id , students.first_name , students.middle_name , students.last_name , student_enrolment.student_id from student_enrolment join students on student_enrolment.student_id = students.student_id group by student_enrolment.student_id having count ( * ) = 2
select students.first_name , students.middle_name , students.last_name , student_enrolment.student_id from student_enrolment join students on student_enrolment.student_id = students.student_id group by student_enrolment.student_id having count ( * ) = 2
select students.first_name , students.middle_name , students.last_name from student_enrolment join students on student_enrolment.student_id = students.student_id join degree_programs on student_enrolment.degree_program_id = degree_programs.degree_program_id where degree_programs.degree_summary_name = 'Bachelor'
select students.first_name , students.middle_name , students.last_name from student_enrolment join students on student_enrolment.student_id = students.student_id where student_enrolment.degree_program_id = 'Bachelors'
select degree_program_id from student_enrolment group by degree_program_id order by count ( * ) desc limit 1
select degree_programs.degree_summary_name from student_enrolment join degree_programs on student_enrolment.degree_program_id = degree_programs.degree_program_id group by student_enrolment.student_id order by count ( * ) desc limit 1
select student_enrolment.student_id , student_enrolment.student_enrolment_id from student_enrolment join degree_programs on student_enrolment.degree_program_id = degree_programs.degree_program_id group by student_enrolment.student_id order by count ( * ) desc limit 1
select degree_programs.degree_program_id , degree_programs.degree_summary_description from degree_programs join student_enrolment on degree_programs.degree_program_id = student_enrolment.degree_program_id group by student_enrolment.degree_program_id order by count ( * ) desc limit 1
select student_enrolment.student_id , students.first_name , students.middle_name , students.last_name , count ( * ) from student_enrolment join students on student_enrolment.student_id = students.student_id group by student_enrolment.student_id order by count ( * ) desc limit 1
select student_enrolment.student_id , students.first_name , students.middle_name , students.last_name , count ( student_enrolment.student_enrolment_id ) from student_enrolment join students on student_enrolment.student_id = students.student_id group by student_enrolment.student_id order by count ( * ) desc limit 1
select semester_name from semesters where semester_id not in ( select semester_id from student_enrolment )
select semester_name from semesters where semester_id not in ( select semester_id from student_enrolment )
select courses.course_name from courses join student_enrolment_courses on courses.course_id = student_enrolment_courses.course_id
select courses.course_name from courses join student_enrolment_courses on courses.course_id = student_enrolment_courses.course_id
select courses.course_name from courses join student_enrolment_courses on courses.course_id = student_enrolment_courses.course_id group by courses.course_name order by count ( * ) desc limit 1
select courses.course_name from courses join student_enrolment_courses on courses.course_id = student_enrolment_courses.course_id group by student_enrolment_courses.course_id order by count ( * ) desc limit 1
sql placeholder
select last_name from students where student_id not in ( select student_id from student_enrolment )
sql placeholder
sql placeholder
select cell_mobile_number from students where first_name = 'Timmothy' and last_name = 'Ward'
select cell_mobile_number from students where first_name = 'Timmothy' and last_name = 'Ward'
select first_name , middle_name , last_name from students order by date_first_registered asc limit 1
select first_name , middle_name , last_name from students order by date_first_registered asc limit 1
select first_name , middle_name , last_name from students order by date_first_registered asc limit 1
select students.first_name , students.middle_name , students.last_name from student_enrolment join students on student_enrolment.student_id = students.student_id order by student_enrolment.degree_program_id asc limit 1
select first_name from students where permanent_address_id != 'null'
select first_name from students where permanent_address_id != 'null'
select addresses.address_id , addresses.line_3 from addresses join students on addresses.address_id = students.current_address_id group by addresses.address_id order by count ( * ) desc limit 1
select addresses.address_id , addresses.line_1 , addresses.line_2 from addresses join students on addresses.address_id = students.permanent_address_id group by addresses.address_id order by count ( * ) desc limit 1
select avg ( transcript_date ) from transcripts
select avg ( transcript_date ) from transcripts
select transcript_date , other_details from transcripts order by transcript_date asc limit 1
select transcript_date , other_details from transcripts order by transcript_date asc limit 1
select count ( * ) from transcripts
select count ( * ) from transcripts
select transcript_date from transcripts order by transcript_date desc limit 1
select transcript_date from transcripts order by transcript_date desc limit 1
select count ( * ) from transcripts group by transcript_id
sql placeholder
select transcript_date , transcript_id from transcripts group by transcript_id order by count ( * ) asc limit 1
select transcript_date , transcript_id from transcripts group by transcript_date order by count ( * ) asc limit 1
select semesters.semester_name from semesters join student_enrolment on semesters.semester_id = student_enrolment.semester_id join degree_programs on student_enrolment.degree_program_id = degree_programs.degree_program_id where degree_programs.degree_summary_name = 'Bachelor'
select semester_id from student_enrolment where semester_id = 'Masters' intersect select semester_id from student_enrolment where student_enrolment.student_id = 'Bachelors'
select count ( distinct addresses.address_id ) from students join addresses on students.current_address_id = addresses.address_id
select distinct addresses.line_3 from addresses join students on addresses.address_id = students.permanent_address_id
select other_student_details from students order by other_student_details desc
select other_student_details from students order by other_student_details desc
select section_description from sections where section_name = 'h'
select section_description from sections where section_name = 'h'
select first_name from students where permanent_address_id = 'Haiti' or cell_mobile_number = '09700166582'
select first_name from students where permanent_address_id = 'Haiti' or cell_mobile_number = '09700166582'
select title from cartoon order by title asc
select title from cartoon order by title asc
select title from cartoon where directed_by = 'Ben Jones'
select title from cartoon where directed_by = 'Ben Jones'
select count ( * ) from cartoon where written_by = 'Joseph Kuhr'
select count ( * ) from cartoon where written_by = 'Joseph Kuhr'
select title , directed_by from cartoon order by original_air_date asc
select title , directed_by from cartoon order by original_air_date asc
select title from cartoon where directed_by = 'Ben Jones' or directed_by = 'Brandon Vietti'
select title from cartoon where directed_by = 'Ben Jones' or directed_by = 'Brandon Vietti'
select country , count ( * ) from tv_channel group by country order by count ( * ) desc limit 1
select country , count ( * ) from tv_channel group by country order by count ( * ) desc limit 1
select count ( distinct series_name ) , count ( distinct content ) from tv_channel
select count ( distinct series_name ) , count ( distinct content ) from tv_channel
select content from tv_channel where series_name = 'Sky Radio'
select content from tv_channel where series_name = 'Sky Radio'
select package_option from tv_channel where series_name = 'Sky Radio'
select package_option from tv_channel where series_name = 'Sky Radio'
select count ( * ) from tv_channel where language = 'English'
select count ( * ) from tv_channel where language = 'English'
select language , count ( * ) from tv_channel group by language order by count ( * ) asc limit 1
select language , count ( * ) from tv_channel group by language order by count ( * ) asc limit 1
select language , count ( * ) from tv_channel group by language
select language , count ( * ) from tv_channel group by language
select tv_channel.series_name from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.title = 'The Rise of the Blue Beetle!'
select tv_channel.series_name from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.title = 'The Rise of the Blue Beetle'
select cartoon.title from tv_channel join cartoon on tv_channel.id = cartoon.channel where tv_channel.series_name = 'Sky Radio'
select cartoon.title from tv_channel join cartoon on tv_channel.id = cartoon.channel where tv_channel.series_name = 'Sky Radio'
select episode from tv_series order by rating asc
select episode from tv_series order by rating asc
select episode , rating from tv_series order by rating desc limit 3
select episode , rating from tv_series order by rating desc limit 3
select min ( share ) , max ( share ) from tv_series
select max ( share ) , min ( share ) from tv_series
select air_date from tv_series where episode = 'A Love of a Lifetime'
select air_date from tv_series where episode = 'A Love of a Lifetime'
select weekly_rank from tv_series where episode = 'A Love of a Lifetime'
select weekly_rank from tv_series where episode = 'A Love of a Lifetime'
select tv_channel.series_name from tv_channel join tv_series on tv_channel.id = tv_series.channel where tv_series.episode = 'A Love of a Lifetime'
select tv_channel.series_name from tv_series join tv_channel on tv_channel.id = tv_series.channel where tv_series.episode = 'A Love of a Lifetime'
select tv_series.episode from tv_series join tv_channel on tv_channel.id = tv_series.channel where tv_channel.series_name = 'Sky Radio'
select tv_series.episode from tv_series join tv_channel on tv_series.channel = tv_channel.id where tv_channel.series_name = 'Sky Radio'
select count ( * ) , directed_by from cartoon group by directed_by
select directed_by , count ( * ) from cartoon group by directed_by
select production_code , channel from cartoon order by original_air_date desc limit 1
select production_code , channel from cartoon order by original_air_date desc limit 1
select package_option , series_name from tv_channel where hight_definition_tv = 'Yes'
select package_option , series_name from tv_channel where hight_definition_tv = 'Yes'
select tv_channel.country from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.written_by = 'Todd Casey'
select tv_channel.country from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.written_by = 'Todd Casey'
select country from tv_channel where tv_channel.country not in ( select tv_channel.country from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.written_by = 'Todd Casey' )
select country from tv_channel except select tv_channel.country from tv_channel join cartoon on tv_channel.id = cartoon.channel where written_by = 'Todd Casey'
select tv_channel.series_name , tv_channel.country from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.directed_by = 'Ben Jones' intersect select tv_channel.series_name , tv_channel.country from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.directed_by = 'Michael Chang'
select tv_channel.series_name , tv_channel.country from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.directed_by = 'Ben Jones' intersect select tv_channel.series_name , tv_channel.country from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.directed_by = 'Michael Chang'
select pixel_aspect_ratio_par , country from tv_channel where language != 'English'
select pixel_aspect_ratio_par , tv_channel.country from tv_channel where language != 'English'
select tv_channel.id from tv_channel join cartoon on tv_channel.id = cartoon.channel group by tv_channel.id having count ( * ) > 2
select id from tv_channel group by id having count ( * ) > 2
select id from tv_channel except select tv_channel.id from tv_channel join cartoon on tv_channel.id = cartoon.channel where directed_by = 'Ben Jones'
select id from tv_channel except select tv_channel.id from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.directed_by = 'Ben Jones'
select package_option from tv_channel except select tv_channel.package_option from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.directed_by = 'Ben Jones'
select package_option from tv_channel except select tv_channel.package_option from tv_channel join cartoon on tv_channel.id = cartoon.channel where cartoon.directed_by = 'Ben Jones'
select count ( * ) from poker_player
select count ( * ) from poker_player
select earnings from poker_player order by earnings desc
select earnings from poker_player order by earnings desc
select final_table_made , best_finish from poker_player
select final_table_made , best_finish from poker_player
select avg ( earnings ) from poker_player
select avg ( earnings ) from poker_player
select money_rank from poker_player order by earnings desc limit 1
select money_rank from poker_player order by earnings desc limit 1
select max ( final_table_made ) from poker_player where earnings < 200000
select max ( final_table_made ) from poker_player where earnings < 200000
select people.name from poker_player join people on poker_player.people_id = people.people_id
select people.name from poker_player join people on poker_player.people_id = people.people_id
select people.name from people join poker_player on people.people_id = poker_player.people_id where poker_player.earnings > 300000
select people.name from people join poker_player on people.people_id = poker_player.people_id where poker_player.earnings > 300000
select people.name from people join poker_player on people.people_id = poker_player.people_id order by final_table_made asc
select people.name from poker_player join people on poker_player.people_id = people.people_id order by final_table_made asc
select people.birth_date from poker_player join people on poker_player.people_id = people.people_id order by poker_player.earnings asc limit 1
select people.birth_date from poker_player join people on poker_player.people_id = people.people_id order by poker_player.earnings asc limit 1
select poker_player.money_rank from poker_player join people on poker_player.people_id = people.people_id order by people.height desc limit 1
select poker_player.money_rank from poker_player join people on poker_player.people_id = people.people_id order by people.height desc limit 1
select avg ( poker_player.earnings ) from poker_player join people on poker_player.people_id = people.people_id where people.height > 200
select avg ( poker_player.earnings ) from poker_player join people on poker_player.people_id = people.people_id where people.height > 200
select people.name from poker_player join people on poker_player.people_id = people.people_id order by poker_player.earnings desc
select people.name from poker_player join people on poker_player.people_id = people.people_id order by poker_player.earnings desc
select nationality , count ( * ) from people group by nationality
select nationality , count ( * ) from people group by nationality
select nationality from people group by nationality order by count ( * ) desc limit 1
select nationality from people group by nationality order by count ( * ) desc limit 1
select nationality from people group by nationality having count ( * ) >= 2
select nationality from people group by nationality having count ( * ) >= 2
select name , birth_date from people order by name asc
select name , birth_date from people order by name asc
select name from people where nationality != 'Russia'
select name from people where nationality != 'Russia'
select name from people where people_id not in ( select people_id from poker_player )
select name from people where people_id not in ( select people_id from poker_player )
select count ( distinct nationality ) from people
select count ( distinct nationality ) from people
select count ( distinct state ) from area_code_state
select contestant_number , contestant_name from contestants order by contestant_name desc
select vote_id , phone_number , state from votes
select max ( area_code ) , min ( area_code ) from area_code_state
select max ( created ) from votes where state = 'CA'
select contestant_name from contestants where contestant_name != 'Jessie Alloway'
select distinct state , created from votes
select contestants.contestant_number , contestants.contestant_name from votes join contestants on votes.contestant_number = contestants.contestant_number group by votes.contestant_number having count ( * ) >= 2
select contestants.contestant_number , contestants.contestant_name from votes join contestants on votes.contestant_number = contestants.contestant_number where votes.created = '%vote_id%' order by votes.vote_id asc limit 1
select count ( * ) from votes where state = 'NY' or state = 'CA'
select count ( * ) from contestants where contestant_number not in ( select contestant_number from votes )
select area_code from area_code_state group by area_code_state.area_code order by count ( * ) desc limit 1
select votes.created , votes.state , votes.phone_number from votes join contestants on votes.contestant_number = contestants.contestant_number where contestants.contestant_name = 'Tabatha Gehling'
sql placeholder
select contestant_name from contestants where contestant_name like '%Al%'
select name from country where indepyear > 1950
select name from country where indepyear > 1950
select count ( * ) from country where governmentform = 'Republic'
select count ( * ) from country where governmentform = 'Republic'
select sum ( surfacearea ) from country where region = 'Caraibes'
select sum ( surfacearea ) from country where region = 'Carribean'
select continent from country where name = 'Anguilla'
select continent from country where name = 'Anguilla'
select country.region from city join country on city.countrycode = country.code where city.name = 'Kabul'
select country.region from city join country on city.countrycode = country.code where city.name = 'Kabul'
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.name = 'Aruba' group by countrylanguage.language order by count ( * ) desc limit 1
select countrylanguage.language from countrylanguage join country on countrylanguage.countrycode = country.code where country.name = 'Aruba' group by countrylanguage.language
select population , lifeexpectancy from country where name = 'Brazil'
select population , lifeexpectancy from country where name = 'Brazil'
select region , population from country where localname = 'Angola'
select region , population from country where name = 'Angola'
select avg ( lifeexpectancy ) from country where region = 'Central Africa'
select avg ( lifeexpectancy ) from country where region = 'Central Africa'
select name from country where lifeexpectancy = 'Asia' order by lifeexpectancy asc limit 1
select name from country where continent = 'Asia' order by lifeexpectancy asc limit 1
select sum ( population ) , max ( gnp ) from country where continent = 'Asia'
select sum ( gnp ) , population from country where continent = 'Asia' group by gnp
select avg ( lifeexpectancy ) from country where governmentform = 'Republic'
select avg ( lifeexpectancy ) from country where governmentform = 'Republic' and continent = 'Africa'
select sum ( surfacearea ) from country where continent = 'Asia' and continent = 'Europe'
select sum ( surfacearea ) from country where continent = 'Asia' or continent = 'Europe'
select count ( * ) from city where district = 'Gelderland'
select sum ( population ) from city where district = 'Gelderland'
select avg ( gnp ) , sum ( population ) from country where governmentform = 'US Territory'
select avg ( gnp ) , sum ( population ) from country where governmentform = 'US Territory'
select count ( distinct language ) from countrylanguage
select count ( distinct language ) from countrylanguage
select count ( * ) from country where continent = 'Africa'
select count ( distinct governmentform ) from country where continent = 'Africa'
select count ( * ) from countrylanguage join country on country.code = countrylanguage.countrycode where country.name = 'Aruba'
select count ( * ) from countrylanguage join country on country.code = countrylanguage.countrycode where country.name = 'Aruba'
select count ( * ) from country where name = 'Afghanistan'
select count ( * ) from country where name = 'Afghanistan'
select country.name from country join countrylanguage on country.code = countrylanguage.countrycode group by countrylanguage.countrycode order by count ( * ) desc limit 1
select country.name from country join countrylanguage on country.code = countrylanguage.countrycode group by countrylanguage.countrycode order by count ( * ) desc limit 1
select continent from country group by continent order by count ( * ) desc limit 1
select continent from country group by continent order by count ( * ) desc limit 1
select count ( * ) from countrylanguage where language = 'English' intersect select count ( * ) from countrylanguage where language = 'Dutch'
select count ( * ) from countrylanguage where language = 'English' intersect select count ( * ) from countrylanguage where language = 'Dutch'
select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'English' intersect select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'French'
select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'English' intersect select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'French'
select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'English' intersect select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'French'
select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'English' intersect select country.name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'French'
select count ( distinct country.continent ) from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'Chinese'
select count ( distinct countrycode ) from countrylanguage where language = 'Chinese'
select country.region from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'English' or countrylanguage.language = 'Dutch'
select country.region from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language = 'Dutch' or countrylanguage.language = 'English'
select countrycode from countrylanguage where language = 'English' or language = 'Dutch'
select countrycode from countrylanguage where language = 'English' or language = 'Dutch'
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.continent = 'Asia' group by countrylanguage.language order by count ( * ) desc limit 1
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.continent = 'Asia' group by countrylanguage.language order by count ( * ) desc limit 1
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.governmentform = 'Republic' group by countrylanguage.language having count ( * ) = 1
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.governmentform = 'Republic' group by countrylanguage.language having count ( * ) = 1
sql placeholder
select city.population from city join countrylanguage on city.countrycode = countrylanguage.countrycode where countrylanguage.language = 'English' order by city.population desc limit 1
select name , population , lifeexpectancy from country where continent = 'Asia' order by surfacearea desc limit 1
select name , population , lifeexpectancy from country where continent = 'Asia'
sql placeholder
sql placeholder
select sum ( country.population ) from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language != 'English'
sql placeholder
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.headofstate = 'Batrix'
select countrylanguage.language from country join countrylanguage on country.code = countrylanguage.countrycode where country.headofstate = 'Batrix'
select count ( distinct countrylanguage.language ) from country join countrylanguage on country.code = countrylanguage.countrycode where country.indepyear < 1930
select count ( distinct language ) from countrylanguage where isofficial < 1930
select name from country where surfacearea > ( select max ( surfacearea ) from country where continent = 'Europe' )
select name from country where surfacearea > ( select max ( surfacearea ) from country where continent = 'Europe' )
select name from country where population < ( select max ( population ) from country where continent = 'Africa' )
select name from country where population < ( select min ( population ) from country where continent = 'Africa' )
select name from city where population > ( select max ( population ) from country where continent = 'Asia' )
select name from city where population > ( select max ( population ) from country where continent = 'Asia' )
select countrycode from countrylanguage where language != 'English'
select countrycode from countrylanguage where language != 'English'
select countrycode from countrylanguage where language != 'English'
select countrycode from countrylanguage where language != 'English'
sql placeholder
sql placeholder
select name from city where countrycode in ( select countrycode from country where continent = 'Europe' )
select name from country join countrylanguage on country.code = countrylanguage.countrycode where countrylanguage.language != 'English'
sql placeholder
select distinct name from country join countrylanguage on country.code = countrylanguage.countrycode where country.continent = 'Asia' and countrylanguage.language = 'Chinese'
select name , indepyear , surfacearea from country order by population asc limit 1
select name , indepyear , surfacearea from country order by population asc limit 1
select city.population , country.name , country.headofstate from city join country on city.countrycode = country.code order by country.surfacearea desc limit 1
select name , population , headofstate from country order by surfacearea desc limit 1
select country.name , count ( * ) from country join countrylanguage on country.code = countrylanguage.countrycode group by countrylanguage.countrycode having count ( * ) >= 3
select country.name , count ( * ) from country join countrylanguage on country.code = countrylanguage.countrycode group by countrylanguage.countrycode having count ( * ) > 2
select count ( * ) , district from city where population > ( select avg ( population ) from city ) group by district
select count ( * ) , district from city where population > ( select avg ( population ) from city ) group by district
select governmentform , sum ( population ) from country group by governmentform having avg ( lifeexpectancy ) >= 72
select governmentform , sum ( population ) from country group by governmentform having avg ( lifeexpectancy ) >= 72
select avg ( lifeexpectancy ) , sum ( population ) , country.continent from country group by country.continent having avg ( lifeexpectancy ) < 72
select continent , sum ( population ) , avg ( lifeexpectancy ) from country group by continent having avg ( lifeexpectancy ) < 72
select name , surfacearea from country order by surfacearea desc limit 5
select name , surfacearea from country order by surfacearea desc limit 5
select name from country order by population desc limit 3
select name from country order by population desc limit 3
select name from country order by population asc limit 3
select name from country order by population asc limit 3
select count ( * ) from country where continent = 'Asia'
select count ( * ) from country where continent = 'Asia'
select name from country where continent = 'Europe' and population = 80000
select name from country where continent = 'Europe' and population = 80000
select sum ( population ) , avg ( surfacearea ) from country where continent = 'North America' and surfacearea > 3000
select sum ( population ) , avg ( surfacearea ) from country where continent = 'North America' and region = 'North America'
select name from city where population between 160000 and 900000
select name from city where population between 160000 and 900000
select language from countrylanguage group by language order by count ( * ) desc limit 1
select language from countrylanguage group by language order by count ( * ) desc limit 1
select language , max ( percentage ) from countrylanguage group by countrycode
select countrylanguage.language , max ( countrylanguage.percentage ) from country join countrylanguage on country.code = countrylanguage.countrycode group by country.code
sql placeholder
select count ( * ) from countrylanguage where language = 'Spanish'
select countrycode from countrylanguage where language = 'Spanish' and percentage = ( select max ( percentage ) from countrylanguage where language = 'Spanish' )
select countrycode from countrylanguage where language = 'Spanish' group by countrycode having count ( * ) >= 1
select count ( * ) from conductor
select count ( * ) from conductor
select name from conductor order by age asc
select name from conductor order by age asc
select name from conductor where nationality != 'USA'
select name from conductor where nationality != 'USA'
select record_company from orchestra order by year_of_founded desc
select record_company from orchestra order by year_of_founded desc
select avg ( attendance ) from show
select avg ( attendance ) from show
select max ( share ) , min ( share ) from performance where type != 'Live final'
select max ( share ) , min ( share ) from performance where type != 'Live final'
select count ( distinct nationality ) from conductor
select count ( distinct nationality ) from conductor
select name from conductor order by year_of_work desc
select name from conductor order by year_of_work desc
select name from conductor order by year_of_work desc limit 1
select name from conductor order by year_of_work desc limit 1
select conductor.name , orchestra.orchestra from conductor join orchestra on conductor.conductor_id = orchestra.conductor_id
select conductor.name , orchestra.orchestra from orchestra join conductor on orchestra.conductor_id = conductor.conductor_id
select conductor.name from orchestra join conductor on orchestra.conductor_id = conductor.conductor_id group by orchestra.conductor_id having count ( * ) > 1
select conductor.name from orchestra join conductor on orchestra.conductor_id = conductor.conductor_id group by orchestra.conductor_id having count ( * ) > 1
select conductor.name from orchestra join conductor on orchestra.conductor_id = conductor.conductor_id group by orchestra.conductor_id order by count ( * ) desc limit 1
select conductor.name from orchestra join conductor on orchestra.conductor_id = conductor.conductor_id group by orchestra.conductor_id order by count ( * ) desc limit 1
select conductor.name from orchestra join conductor on orchestra.conductor_id = conductor.conductor_id where orchestra.year_of_founded > 2008
select conductor.name from orchestra join conductor on orchestra.conductor_id = conductor.conductor_id where orchestra.year_of_founded > 2008
select record_company , count ( * ) from orchestra group by record_company
select record_company , count ( * ) from orchestra group by record_company
select major_record_format from orchestra group by major_record_format order by count ( * ) asc
select major_record_format from orchestra group by major_record_format order by count ( * ) asc
select record_company from orchestra group by record_company order by count ( * ) desc limit 1
select record_company from orchestra group by record_company order by count ( * ) desc limit 1
select orchestra from orchestra where orchestra_id not in ( select orchestra_id from performance )
select orchestra from orchestra where orchestra_id not in ( select orchestra_id from performance )
select record_company from orchestra where year_of_founded < 2003 intersect select record_company from orchestra where year_of_founded > 2003
select record_company from orchestra where year_of_founded < 2003 intersect select record_company from orchestra where year_of_founded > 2003
select count ( * ) from orchestra where major_record_format = 'CD' or major_record_format = 'DVD'
select count ( * ) from orchestra where major_record_format = 'CD' or major_record_format = 'DVD'
select year_of_founded from orchestra group by year_of_founded having count ( * ) > 1
select year_of_founded from orchestra group by year_of_founded having count ( * ) > 1
select count ( * ) from highschooler
select count ( * ) from highschooler
select name , grade from highschooler
select name , grade from highschooler
select grade from highschooler
select grade from highschooler
select grade from highschooler where name = 'Kyle'
select grade from highschooler where name = 'Kyle'
select name from highschooler where grade = 10
select name from highschooler where grade = 10
select id from highschooler where name = 'Kyle'
select id from highschooler where name = 'Kyle'
select count ( * ) from highschooler where grade = 9 or grade = 10
select count ( * ) from highschooler where grade = 9 or grade = 10
select count ( * ) , grade from highschooler group by grade
select count ( * ) , grade from highschooler group by grade
select grade from highschooler group by grade order by count ( * ) desc limit 1
select grade from highschooler group by grade order by count ( * ) desc limit 1
select grade from highschooler group by grade having count ( * ) >= 4
select grade from highschooler group by grade having count ( * ) >= 4
select student_id , count ( * ) from friend group by student_id
select count ( * ) , student_id from friend group by student_id
select highschooler.name , count ( * ) from highschooler join friend on highschooler.id = friend.student_id group by friend.student_id
select highschooler.name , count ( * ) from highschooler join friend on highschooler.id = friend.student_id group by highschooler.id
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id group by friend.student_id order by count ( * ) desc limit 1
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id group by friend.student_id order by count ( * ) desc limit 1
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id group by friend.student_id having count ( * ) >= 3
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id group by friend.student_id having count ( * ) >= 3
select friend.friend_id from highschooler join friend on highschooler.id = friend.student_id where highschooler.name = 'Kyle'
select friend.friend_id from highschooler join friend on highschooler.id = friend.student_id where highschooler.name = 'Kyle'
select count ( * ) from highschooler join friend on highschooler.id = friend.student_id where highschooler.name = 'Kyle'
select count ( * ) from highschooler where name = 'Kyle'
sql placeholder
select id from highschooler except select student_id from friend
select name from highschooler where id not in ( select student_id from friend )
select name from highschooler where id not in ( select student_id from friend )
select friend.friend_id from highschooler join friend on highschooler.id = friend.friend_id intersect select likes.liked_id from highschooler join likes on highschooler.id = likes.liked_id where highschooler.name = 'H'
sql placeholder
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id join likes on highschooler.id = likes.liked_id where friend.friend_id = 'friend' intersect select highschooler.name from highschooler join likes on highschooler.id = likes.liked_id where likes.liked_id = 'null'
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id join likes on highschooler.id = likes.liked_id where friend.friend_id = 'highschooler.id' intersect select highschooler.name from highschooler join likes on highschooler.id = likes.liked_id join friend on highschooler.id = friend.student_id where friend.friend_id = 'highschooler.id'
select count ( * ) , student_id from likes group by student_id
select student_id , count ( * ) from likes group by student_id
select highschooler.name , count ( * ) from highschooler join likes on highschooler.id = likes.student_id group by likes.student_id
select highschooler.name , count ( * ) from highschooler join likes on highschooler.id = likes.student_id group by likes.student_id
select highschooler.name from highschooler join likes on highschooler.id = likes.student_id group by likes.student_id order by count ( * ) desc limit 1
select highschooler.name from highschooler join likes on highschooler.id = likes.student_id group by likes.student_id order by count ( * ) desc limit 1
select highschooler.name from highschooler join likes on highschooler.id = likes.student_id group by likes.student_id having count ( * ) >= 2
select highschooler.name from highschooler join likes on highschooler.id = likes.student_id group by likes.student_id having count ( * ) >= 2
select highschooler.name from highschooler join friend on highschooler.id = friend.student_id where highschooler.grade > 5 group by friend.student_id having count ( * ) >= 2
select name from highschooler where grade > 5 group by name having count ( * ) >= 2
select count ( * ) from highschooler where name = 'Kyle'
select count ( * ) from highschooler join likes on highschooler.id = likes.student_id where highschooler.name = 'Kyle'
select avg ( highschooler.grade ) from highschooler join friend on highschooler.id = friend.student_id where friend.friend_id != 'null'
select avg ( grade ) from highschooler join friend on highschooler.id = friend.student_id
select min ( grade ) from highschooler where id not in ( select student_id from friend )
select min ( grade ) from highschooler where id not in ( select student_id from friend )
select state from owners intersect select state from professionals
select state from owners intersect select state from professionals
select avg ( dogs.age ) from dogs join treatments on dogs.dog_id = treatments.dog_id where dogs.age in ( select dogs.age from treatments join dogs on treatments.dog_id = dogs.dog_id )
select avg ( dogs.age ) from dogs join treatments on dogs.dog_id = treatments.dog_id
select professionals.professional_id , professionals.last_name , professionals.cell_number from professionals join treatments on professionals.professional_id = treatments.professional_id where professionals.state = 'Indiana' group by professionals.professional_id having count ( * ) > 2
select professionals.professional_id , professionals.last_name , professionals.cell_number from professionals join treatments on professionals.professional_id = treatments.professional_id where professionals.state = 'Indiana' union select treatments.professional_id , professionals.last_name , professionals.cell_number from professionals join treatments on professionals.professional_id = treatments.professional_id where professionals.state = 'Indiana'
select dogs.name from dogs join treatments on dogs.dog_id = treatments.dog_id where treatments.cost_of_treatment not in ( select dog_id from treatments where cost_of_treatment > 1000 )
select dogs.name from dogs join treatments on dogs.owner_id = treatments.dog_id where treatments.cost_of_treatment not in ( select dogs.owner_id from owners join treatments on owners.owner_id = treatments.dog_id where treatments.cost_of_treatment > 1000 )
select first_name from professionals where role_code = 'professional' except select owners.first_name from owners join dogs on owners.owner_id = dogs.owner_id where dogs.name = 'dog'
select first_name from professionals where role_code = 'PRO' except select dogs.name from owners join dogs on owners.owner_id = dogs.owner_id join professionals on owners.owner_id = professionals.professional_id where professionals.first_name = 'PRO'
select professional_id , role_code , email_address from professionals except select professionals.professional_id , professionals.role_code , professionals.email_address from professionals join treatments on professionals.professional_id = treatments.professional_id join dogs on treatments.dog_id = dogs.dog_id
select professional_id , role_code , email_address from professionals except select professionals.professional_id , professionals.role_code , professionals.email_address from professionals join treatments on professionals.professional_id = treatments.professional_id join dogs on treatments.dog_id = dogs.dog_id
select owners.owner_id , owners.first_name , owners.last_name from owners join dogs on owners.owner_id = dogs.owner_id group by owners.owner_id order by count ( * ) desc limit 1
select owners.owner_id , owners.first_name , owners.last_name from owners join dogs on owners.owner_id = dogs.owner_id group by dogs.owner_id order by count ( * ) desc limit 1
select professionals.professional_id , professionals.role_code , professionals.first_name from professionals join treatments on professionals.professional_id = treatments.professional_id group by professionals.professional_id having count ( * ) >= 2
select professionals.professional_id , professionals.role_code , professionals.first_name from professionals join treatments on professionals.professional_id = treatments.professional_id group by professionals.professional_id having count ( * ) >= 2
select breeds.breed_name from breeds join dogs on breeds.breed_code = dogs.breed_code group by breeds.breed_code order by count ( * ) desc limit 1
select breeds.breed_name from breeds join dogs on breeds.breed_code = dogs.breed_code group by breeds.breed_name order by count ( * ) desc limit 1
select owners.owner_id , owners.last_name from owners join treatments on owners.owner_id = treatments.dog_id group by owners.owner_id order by sum ( treatments.cost_of_treatment ) desc limit 1
select owners.owner_id , owners.last_name from owners join treatments on owners.owner_id = treatments.dog_id group by owners.owner_id order by sum ( treatments.cost_of_treatment ) desc limit 1
select treatment_types.treatment_type_description from treatments join treatment_types on treatments.treatment_type_code = treatment_types.treatment_type_code group by treatments.treatment_type_code order by sum ( treatments.cost_of_treatment ) asc limit 1
select treatment_types.treatment_type_description from treatments join treatment_types on treatments.treatment_type_code = treatment_types.treatment_type_code group by treatments.treatment_type_code order by sum ( cost_of_treatment ) asc limit 1
sql placeholder
sql placeholder
select professionals.professional_id , professionals.cell_number from professionals join treatments on professionals.professional_id = treatments.professional_id group by professionals.professional_id having count ( * ) >= 2
select professionals.professional_id , professionals.cell_number from professionals join treatments on professionals.professional_id = treatments.professional_id group by professionals.professional_id having count ( * ) >= 2
select professionals.first_name , professionals.last_name from professionals join treatments on professionals.professional_id = treatments.professional_id where treatments.cost_of_treatment < ( select avg ( cost_of_treatment ) from treatments )
select professionals.first_name , professionals.last_name from professionals join treatments on professionals.professional_id = treatments.professional_id where treatments.cost_of_treatment < ( select avg ( cost_of_treatment ) from treatments )
select date_of_treatment , professionals.first_name , professionals.last_name from treatments join professionals on treatments.professional_id = professionals.professional_id
select date_of_treatment , professionals.first_name from treatments join professionals on treatments.professional_id = professionals.professional_id
select cost_of_treatment , treatment_types.treatment_type_description from treatments join treatment_types on treatments.treatment_type_code = treatment_types.treatment_type_code
select treatments.cost_of_treatment , treatment_types.treatment_type_description from treatments join treatment_types on treatments.treatment_type_code = treatment_types.treatment_type_code
select owners.first_name , owners.last_name , sizes.size_description from owners join dogs on owners.owner_id = dogs.owner_id join sizes on dogs.size_code = sizes.size_code
select owners.first_name , owners.last_name , dogs.size_code from owners join dogs on owners.owner_id = dogs.owner_id join sizes on dogs.size_code = sizes.size_code
select owners.first_name , dogs.name from owners join dogs on owners.owner_id = dogs.owner_id
select owners.first_name , dogs.name from owners join dogs on owners.owner_id = dogs.owner_id
select dogs.name , treatments.date_of_treatment from dogs join treatments on dogs.dog_id = treatments.dog_id join breeds on dogs.breed_code = breeds.breed_code order by breeds.breed_name asc limit 1
select dogs.name , treatments.date_of_treatment from dogs join treatments on dogs.dog_id = treatments.dog_id join breeds on dogs.breed_code = breeds.breed_code where breeds.breed_name = ( select max ( date_of_treatment ) from treatments )
select dogs.name , owners.first_name from owners join dogs on owners.owner_id = dogs.owner_id where owners.state = 'Virginia'
select owners.first_name , dogs.name from owners join dogs on owners.owner_id = dogs.owner_id where owners.state = 'Virginia'
select dogs.date_arrived , dogs.date_departed from dogs join treatments on dogs.dog_id = treatments.dog_id
select dogs.date_arrived , dogs.date_departed from dogs join treatments on dogs.dog_id = treatments.dog_id
select owners.last_name from owners join dogs on owners.owner_id = dogs.owner_id order by dogs.age asc limit 1
select owners.last_name from owners join dogs on owners.owner_id = dogs.owner_id order by dogs.age asc limit 1
select email_address from professionals where state = 'Hawaii' or state = 'Wisconsin'
select email_address from professionals where state = 'Hawaii' or state = 'Wisconsin'
select date_arrived , date_departed from dogs
select date_arrived , date_departed from dogs
select count ( * ) from treatments join dogs on treatments.dog_id = dogs.dog_id
select count ( distinct dog_id ) from treatments
select count ( distinct professional_id ) from treatments
select count ( distinct professional_id ) from treatments
select role_code , street , state from professionals where city like '%West%'
select role_code , street , city , state from professionals where city like '%West%'
select first_name , last_name , email_address from owners where state like '%North%'
select first_name , last_name , email_address from owners where state like '%North%'
select count ( * ) from dogs where age < ( select avg ( age ) from dogs )
select count ( * ) from dogs where age < ( select avg ( age ) from dogs )
select cost_of_treatment from treatments order by date_of_treatment desc limit 1
select cost_of_treatment from treatments order by date_of_treatment desc limit 1
select count ( * ) from dogs where dog_id not in ( select dog_id from treatments )
select count ( * ) from dogs where dog_id not in ( select dog_id from treatments )
select count ( * ) from owners where owner_id not in ( select owner_id from dogs )
select count ( * ) from owners where owner_id not in ( select owner_id from dogs )
select count ( * ) from professionals where professional_id not in ( select professional_id from treatments )
select count ( * ) from professionals where professional_id not in ( select professional_id from treatments )
select name , age , weight from dogs where abandoned_yn = 1
select name , age , weight from dogs where abandoned_yn = 1
select avg ( age ) from dogs
select avg ( age ) from dogs
select age from dogs order by age desc limit 1
select age from dogs order by age desc limit 1
select charge_type , sum ( charge_amount ) from charges group by charge_type
select charge_type , sum ( charge_amount ) from charges group by charge_type
select charge_type from charges order by charge_amount desc limit 1
select charge_amount from charges order by charge_type desc limit 1
select email_address , cell_number , home_phone from professionals
select email_address , cell_number , home_phone from professionals
select breeds.breed_name , sizes.size_description from breeds join sizes on breeds.breed_code = sizes.size_code union select breeds.breed_name , sizes.size_description from breeds join sizes on breeds.breed_code = sizes.size_code
select distinct breeds.breed_name , sizes.size_description from breeds join dogs on breeds.breed_code = dogs.breed_code join sizes on dogs.size_code = sizes.size_code
select professionals.first_name , treatment_types.treatment_type_description from treatments join professionals on treatments.professional_id = professionals.professional_id join treatment_types on treatments.treatment_type_code = treatment_types.treatment_type_code
select professionals.first_name , professionals.last_name , treatment_types.treatment_type_description from professionals join treatments on professionals.professional_id = treatments.professional_id join treatment_types on treatments.treatment_type_code = treatment_types.treatment_type_code
select count ( * ) from singer
select count ( * ) from singer
select name from singer order by net_worth_millions asc
select name from singer order by net_worth_millions asc
select birth_year , citizenship from singer
select birth_year , citizenship from singer
select name from singer where citizenship != 'France'
select name from singer where citizenship != 'France'
select name from singer where birth_year = 1948 or birth_year = 1949
select name from singer where birth_year = 1948 or birth_year = 1949
select name from singer order by net_worth_millions desc limit 1
select name from singer order by net_worth_millions desc limit 1
select citizenship , count ( * ) from singer group by citizenship
select citizenship , count ( * ) from singer group by citizenship
select citizenship from singer group by citizenship order by count ( * ) desc limit 1
select citizenship from singer group by citizenship order by count ( * ) desc limit 1
select citizenship , max ( net_worth_millions ) from singer group by citizenship
select citizenship , max ( net_worth_millions ) from singer group by citizenship
select song.title , singer.name from song join singer on song.singer_id = singer.singer_id
select song.title , singer.name from song join singer on song.singer_id = singer.singer_id
select distinct singer.name from singer join song on singer.singer_id = song.singer_id where song.sales > 300000
select distinct name from singer join song on singer.singer_id = song.singer_id where song.sales > 300000
select singer.name from singer join song on singer.singer_id = song.singer_id group by song.singer_id having count ( * ) > 1
select singer.name from singer join song on singer.singer_id = song.singer_id group by song.singer_id having count ( * ) > 1
select singer.name , sum ( song.sales ) from song join singer on song.singer_id = singer.singer_id group by song.singer_id
select singer.name , sum ( song.sales ) from singer join song on singer.singer_id = song.singer_id group by song.singer_id
select name from singer where singer_id not in ( select singer_id from song )
select name from singer where singer_id not in ( select singer_id from song )
select citizenship from singer where birth_year < 1945 intersect select citizenship from singer where birth_year > 1955
select citizenship from singer where birth_year < 1945 intersect select citizenship from singer where birth_year > 1955
select count ( * ) from other_available_features
select ref_feature_types.feature_type_name from ref_feature_types join other_available_features on ref_feature_types.feature_type_code = other_available_features.feature_id where other_available_features.feature_name = 'AirCon'
select property_type_description from properties join ref_property_types on properties.property_type_code = ref_property_types.property_type_code
select property_name from properties where property_type_code = 'House' or property_type_code = 'Apartment' and room_count > 1