#1
import csv

# Define the list to store users from London
users_in_london = []

# Read the CSV file with UTF-8 encoding
with open('users.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        location = row['location'].strip().lower()
        # Check if the user is from london
        if 'london' in location:
            users_in_london.append({
                'login': row['login'],
                'followers': int(row['followers'])
            })

# Sort users based on followers in descending order
top_users = sorted(users_in_london, key=lambda x: x['followers'], reverse=True)

# Extract the top 5 user logins
top_5_logins = [user['login'] for user in top_users[:5]]

# Print the result as a comma-separated list
print(','.join(top_5_logins))


#2

import csv
from datetime import datetime

# Define the list to store users from London
users_in_london = []

# Read the CSV file with UTF-8 encoding
with open('users.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        location = row['location'].strip().lower()
        # Check if the user is from Delhi
        if 'london' in location:
            users_in_london.append({
                'login': row['login'],
                'created_at': datetime.strptime(row['created_at'], '%Y-%m-%dT%H:%M:%SZ')
            })

# Sort users based on created_at in ascending order
sorted_users = sorted(users_in_london, key=lambda x: x['created_at'])

# Extract the top 5 user logins
top_5_earliest_logins = [user['login'] for user in sorted_users[:5]]

# Print the result as a comma-separated list
print(','.join(top_5_earliest_logins))


#3
import csv
from collections import Counter

# Define the list to store license names
licenses = []

# Read the CSV file with UTF-8 encoding
with open('repositories.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Check if the license_name field is present and not empty
        license_name = row.get('license_name', '').strip()
        if license_name:
            licenses.append(license_name)

# Count the occurrence of each license
license_counts = Counter(licenses)

# Get the 3 most common licenses
top_3_licenses = [license for license, count in license_counts.most_common(3)]

# Print the result as a comma-separated list
print(','.join(top_3_licenses))


#4
import csv
from collections import Counter

# Define the list to store company names
companies = []

# Read the CSV file with UTF-8 encoding
with open('users.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Get and clean up the company field (ignore empty values)
        company = row.get('company', '').strip()
        if company:
            companies.append(company)

# Count the occurrence of each company
company_counts = Counter(companies)

# Find the most common company
most_common_company = company_counts.most_common(1)

# Print the result
if most_common_company:
    print(most_common_company[0][0])
else:
    print("No company data found.")


#5
import csv
from collections import Counter

# Define the list to store programming languages
languages = []

# Read the CSV file with UTF-8 encoding
with open('repositories.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Get and clean up the language field (ignore empty values)
        language = row.get('language', '').strip()
        if language:
            languages.append(language)

# Count the occurrence of each language
language_counts = Counter(languages)

# Find the most common language
most_common_language = language_counts.most_common(1)

# Print the result
if most_common_language:
    print(most_common_language[0][0])
else:
    print("No language data found.")

#6
import csv
from collections import Counter
from datetime import datetime

# Define the list to store programming languages
languages = []

# Read the CSV file with UTF-8 encoding
with open('repositories.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    # Iterate through the rows in the CSV
    for row in reader:
        # Parse the created_at field
        created_at = row.get('created_at', '').strip()
        
        # Convert the date string to a datetime object
        if created_at:
            user_join_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
            
            # Check if the user joined after 2020
            if user_join_date.year > 2020:
                # Get the language field and clean it up
                language = row.get('language', '').strip()
                if language:
                    languages.append(language)

# Count the occurrence of each language
language_counts = Counter(languages)

# Find the two most common languages
most_common_languages = language_counts.most_common(2)

# Print the second most common language
if len(most_common_languages) >= 2:
    print(most_common_languages[1][0])  # Second most common language
else:
    print("Not enough language data found.")

#7
import csv
from collections import defaultdict

# Define a dictionary to store total stars and repository count per language
language_stats = defaultdict(lambda: {'stars': 0, 'repos': 0})

# Read the CSV file with UTF-8 encoding
with open('repositories.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        # Get the language and stargazers_count field
        language = row.get('language', '').strip()
        stars = row.get('stargazers_count', '0').strip()

        # Only process if language and stars are available
        if language and stars.isdigit():
            language_stats[language]['stars'] += int(stars)
            language_stats[language]['repos'] += 1

# Calculate average stars for each language
average_stars_per_language = {
    language: stats['stars'] / stats['repos']
    for language, stats in language_stats.items()
    if stats['repos'] > 0
}

# Find the language with the highest average stars
if average_stars_per_language:
    most_popular_language = max(average_stars_per_language, key=average_stars_per_language.get)
    print(most_popular_language)
else:
    print("No language data found.")

#8
import csv

# Define a list to store users and their leader strength
leader_strengths = []

# Read the CSV file with UTF-8 encoding
with open('users.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        # Get followers and following counts
        followers = int(row.get('followers', 0).strip())
        following = int(row.get('following', 0).strip())
        
        # Calculate leader strength
        leader_strength = followers / (1 + following)
        
        # Store the user's login and their leader strength
        leader_strengths.append((row.get('login', ''), leader_strength))

# Sort users by leader strength in descending order
leader_strengths.sort(key=lambda x: x[1], reverse=True)

# Get the top 5 users
top_5_leaders = [login for login, strength in leader_strengths[:5]]

# Print the result as a comma-separated list
print(','.join(top_5_leaders))

#9
import csv
import numpy as np

# Lists to store the followers and public repos of users from london
followers = []
public_repos = []

# Open the users.csv file and read data
with open('users.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        # Filter for users in london
        location = row.get('location', '').strip().lower()
        if "london" in location:
            # Get followers and public repositories values
            try:
                followers_count = int(row['followers'])
                public_repos_count = int(row['public_repos'])
                
                # Append the valid values to the lists
                followers.append(followers_count)
                public_repos.append(public_repos_count)
            except ValueError:
                # Skip rows with invalid numerical values
                continue

# Ensure there is data to compute correlation
if len(followers) > 1 and len(public_repos) > 1:
    # Compute Pearson correlation coefficient
    correlation_matrix = np.corrcoef(followers, public_repos)
    correlation = correlation_matrix[0, 1]
    # Output correlation rounded to 3 decimal places
    print(f"{correlation:.3f}")
else:
    print("Insufficient data for correlation calculation.")


#10
import csv
import numpy as np

# Lists to store the followers and public repos of users from london
followers = []
public_repos = []

# Open the users.csv file and read data
with open('users.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        # Filter for users in london
        location = row.get('location', '').strip().lower()
        if "london" in location:
            # Get followers and public repositories values
            try:
                followers_count = int(row['followers'])
                public_repos_count = int(row['public_repos'])
                
                # Append the valid values to the lists
                followers.append(followers_count)
                public_repos.append(public_repos_count)
            except ValueError:
                # Skip rows with invalid numerical values
                continue

# Ensure there is data for regression
if len(followers) > 1 and len(public_repos) > 1:
    # Perform linear regression: followers ~ public_repos
    slope, intercept = np.polyfit(public_repos, followers, 1)
    
    # Output the slope rounded to 3 decimal places
    print(f"{slope:.3f}")
else:
    print("Insufficient data for regression.")


#11
import pandas as pd
import numpy as np

def analyze_repo_features(csv_file):
    
    df = pd.read_csv(csv_file)
    
    if df['has_projects'].dtype == 'object':
        df['has_projects'] = df['has_projects'].map({'true': True, 'false': False})
    if df['has_wiki'].dtype == 'object':
        df['has_wiki'] = df['has_wiki'].map({'true': True, 'false': False})
    
    correlation = df['has_projects'].corr(df['has_wiki'])
    
    stats = {
        'total_repos': len(df),
        'projects_enabled': df['has_projects'].sum(),
        'wiki_enabled': df['has_wiki'].sum(),
        'both_enabled': ((df['has_projects']) & (df['has_wiki'])).sum(),
        'neither_enabled': ((~df['has_projects']) & (~df['has_wiki'])).sum()
    }
    
    return round(correlation, 3), stats

correlation, stats = analyze_repo_features('repositories.csv')
print(f"Correlation coefficient: {correlation}")
print("\nAdditional Statistics:")
for key, value in stats.items():
    print(f"{key}: {value}")


#12import pandas as pd

def analyze_following_difference(users_csv_path='users.csv'):
    # Read the data
    df = pd.read_csv(users_csv_path)
    
    # Calculate average following for hireable users
    hireable_following = df[df['hireable'] == True]['following'].mean()
    
    # Calculate average following for non-hireable users
    non_hireable_following = df[df['hireable'] != True]['following'].mean()
    
    # Calculate the difference rounded to 3 decimal places
    difference = round(hireable_following - non_hireable_following, 3)
    
    # Print debug information
    print(f"Number of hireable users: {len(df[df['hireable'] == True])}")
    print(f"Number of non-hireable users: {len(df[df['hireable'] != True])}")
    print(f"Average following for hireable users: {hireable_following:.3f}")
    print(f"Average following for non-hireable users: {non_hireable_following:.3f}")
    
    return difference

# Calculate the difference
result = analyze_following_difference()
print(f"\nDifference in average following: {result:.3f}")

#13
import pandas as pd
import statsmodels.api as sm

# Load the users data from the CSV file
users_df = pd.read_csv('users.csv')

# Filter out users without bios
users_with_bios = users_df[users_df['bio'].notna()]

# Calculate the length of the bio in words
#users_with_bios['bio_word_count'] = users_with_bios['bio'].str.split(" ").str.len()

# The error was here: users_with_bio was used instead of users_with_bios
users_with_bios['bio_word_count'] = users_with_bios['bio'].apply(lambda x: len(x.split()))


# Prepare the data for regression
X = users_with_bios['bio_word_count']  # Independent variable
y = users_with_bios['followers']        # Dependent variable

# Add a constant to the independent variable for the regression
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Get the regression slope (coefficient for bio_word_count)
slope = model.params['bio_word_count']

# Print the slope rounded to three decimal places
print(f'Regression slope of followers on bio word count: {slope:.3f}')


#14
import csv
from collections import Counter
from datetime import datetime

# Counter to store the number of repositories created by each user on weekends
weekend_repo_counts = Counter()

# Open the repositories.csv file and read data
with open('repositories.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        created_at = row.get('created_at', '')
        if created_at:
            # Convert created_at string to a datetime object
            created_date = datetime.fromisoformat(created_at[:-1])  # Remove 'Z' and convert
            
            # Check if the day is Saturday (5) or Sunday (6)
            if created_date.weekday() in [5, 6]:
                user_login = row['login']
                weekend_repo_counts[user_login] += 1  # Increment the count for the user

# Get the top 5 users who created the most repositories on weekends
top_users = weekend_repo_counts.most_common(5)

# Extract the logins of the top users
top_logins = [user[0] for user in top_users]

# Output the top users' logins as a comma-separated string
print(','.join(top_logins))


#15
import pandas as pd

def analyze_email_sharing(users_csv_path='users.csv'):
    # Read the complete CSV file
    df = pd.read_csv(users_csv_path)
    
    # Convert email column to boolean (True if email exists, False if NaN or empty)
    df['has_email'] = df['email'].notna() & (df['email'] != '')
    
    # Calculate for hireable users
    hireable_mask = df['hireable'] == True
    if hireable_mask.any():
        hireable_email_fraction = df[hireable_mask]['has_email'].mean()
    else:
        hireable_email_fraction = 0
        
    # Calculate for non-hireable users
    non_hireable_mask = df['hireable'] != True
    if non_hireable_mask.any():
        non_hireable_email_fraction = df[non_hireable_mask]['has_email'].mean()
    else:
        non_hireable_email_fraction = 0
    
    # Calculate difference and round to 3 decimal places
    difference = round(hireable_email_fraction - non_hireable_email_fraction, 3)
    
    # Print debug information
    print(f"Total users: {len(df)}")
    print(f"Hireable users with email: {df[hireable_mask]['has_email'].sum()}/{hireable_mask.sum()}")
    print(f"Non-hireable users with email: {df[non_hireable_mask]['has_email'].sum()}/{non_hireable_mask.sum()}")
    print(f"Hireable fraction: {hireable_email_fraction:.3f}")
    print(f"Non-hireable fraction: {non_hireable_email_fraction:.3f}")
    
    return difference

# Read and analyze the complete dataset
result = analyze_email_sharing()
print(f"\nFinal result: {result:.3f}")


#16
import csv
from collections import Counter

# Counter to store surname frequencies
surname_counter = Counter()

# Open the users.csv file and read data
with open('users.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        name = row.get('name', '').strip()
        if name:  # Ignore missing names
            # Split the name by whitespace and get the last word as the surname
            surname = name.split()[-1]
            surname_counter[surname] += 1

# Find the maximum frequency of surnames
if surname_counter:
    max_count = max(surname_counter.values())
    # Get all surnames with the maximum frequency
    most_common_surnames = [surname for surname, count in surname_counter.items() if count == max_count]
    # Sort surnames alphabetically
    most_common_surnames.sort()
    # Output the result
    print(f"{','.join(most_common_surnames)}: {max_count}")
else:
    print("No names found.")
