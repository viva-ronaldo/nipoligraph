import os, re, requests, time, yaml
from bs4 import BeautifulSoup
import pandas as pd

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_dir = f"s3://{config['NIPOL_DATA_BUCKET']}/"

# i) Committee meeting agendas and attendances

print('- Getting committee agendas and attendance')

committee_agenda_filepath = os.path.join(data_dir, 'committee_meetings_agendas_feb2024topresent.csv')
committee_attendance_filepath = os.path.join(data_dir, 'committee_meetings_attendances_feb2024topresent.csv')

existing_agenda = pd.read_csv(committee_agenda_filepath)
existing_attendance = pd.read_csv(committee_attendance_filepath)

n_existing_meeting_ids = existing_agenda.meeting_id.nunique()

#start_mid, end_mid = 16100, 17100  # one time, get all from Feb 2024 to present (Jun 2024)
start_mid, end_mid = existing_agenda.meeting_id.max()-50, existing_agenda.meeting_id.max()+25
# use latest that we have; the ids are not quite ordered by date, but close

agenda_res, attendance_res = [], []
for mid in range(start_mid, end_mid+1):
    try:
        page = requests.get(f'https://aims.niassembly.gov.uk/committees/meetings.aspx?&cid=118&md=19/06/2024%2000:00:00&mid={mid}')
    except requests.exceptions.SSLError:
        print(f'SSLError for mid={mid}; skipping')
        continue
    soup = BeautifulSoup(page.text, features='lxml')
    if soup.find(name='h1').text != '':
        committee_name = soup.find(name='h1').text.strip()
        meeting_date = soup.find(name='label').findParent().text[-10:]
        agenda_items = [el.find_all('td')[0].text.strip() for el in soup.find(id='ctl00_MainContentPlaceHolder_AgendaPane1_content').find_all(name='tr')[1:]]
        if len(agenda_items) == 0:
            continue
        if soup.find(id='ctl00_MainContentPlaceHolder_AttendancePane1_content') is None:
            continue
        attendance_list = soup.find(id='ctl00_MainContentPlaceHolder_AttendancePane1_content').find_all(name='tr')[1:]
        attendance_list = [(el.find(name='td').text, 'checked' in el.find(name='input').attrs.keys()) for el in attendance_list]

        agenda_res.append(pd.DataFrame({
            'meeting_id': [mid]*len(agenda_items),
            'committee_name': [committee_name]*len(agenda_items),
            'meeting_date': [meeting_date]*len(agenda_items),
            'agenda_item': agenda_items
            }))
        attendance_res.append(pd.DataFrame({
            'meeting_id': [mid]*len(attendance_list),
            'committee_name': [committee_name]*len(attendance_list),
            'meeting_date': [meeting_date]*len(attendance_list),
            'member': [el[0] for el in attendance_list],
            'attended': [el[1] for el in attendance_list]
            }))

agenda_res = pd.concat(agenda_res)
attendance_res = pd.concat(attendance_res)

existing_agenda = pd.concat([existing_agenda, agenda_res]).drop_duplicates()
existing_attendance = pd.concat([existing_attendance, attendance_res])
existing_attendance['member'] = existing_attendance['member'].str.replace('Mr |Mrs |Ms |Miss |Dr |Sir | OBE| MBE| CBE| MC', '', regex=True)
existing_attendance = existing_attendance.drop_duplicates()
# make it match mla_ids MemberFirstName + ' ' + MemberLastName

n_meeting_ids = existing_agenda.meeting_id.nunique()
print(f'\nFound {n_meeting_ids - n_existing_meeting_ids} new committee meetings with agendas, attendance_lists; have {n_meeting_ids} total')

existing_agenda.to_csv(committee_agenda_filepath, index=False)
existing_attendance.to_csv(committee_attendance_filepath, index=False)

# NB: Chairpersons' Liaison Group has no attendee list

# agenda items are also links, but many contain no information
#https://aims.niassembly.gov.uk/committees/meetingiob.aspx?&cid=12&caid=33426&md=27/06/2024%2000:00:00&mid=16992&iobid=264847&eveid=16992&bd=0


# ii) All-Party Group memberships

print('- Getting All-Party Group memberships')

apg_membership_filepath = os.path.join(data_dir, 'current_apg_group_memberships.csv')

page = requests.get(f'https://aims.niassembly.gov.uk/mlas/allpartygroups.aspx')
soup = BeautifulSoup(page.text, features='lxml')

# We can identify the current APG names but the links are not readable
current_apg_names = [el.text for el in soup.find(id='ctl00_MainContentPlaceHolder_APGGridView').find_all(name='a')]

# Hard code the links to membership pages in July 2024
apg_links_dict = {
    'All-Party Group on Access to Justice': 2185,
    'All-Party Group on Active Travel': 898,
    'All-Party Group on Addiction and Dual Diagnosis': 1721,
    'All-Party Group on ADHD': 1538,
    'All-Party Group on Aerospace, Defence, Security & Space': 2247,
    'All-Party Group on Ageing and Older People': 1360,
    'All-Party Group on Animal Welfare': 1782,
    'All-Party Group on Arts': 1894,
    'All-Party Group on Autism': 227,
    'All-Party Group on Built Environment': 274,
    'All-Party Group on Cancer': 273,
    'All-Party Group on Carers': 1779,
    'All-Party Group on Climate Action': 1631,
    'All-Party Group on Community Pharmacy': 1833,
    'All-Party Group on Dementia': 2196,
    'All-Party Group on Diabetes': 2002,
    'All-Party Group on Disability': 269,
    'All-Party Group on Domestic and Sexual Violence': 1554,
    'All-Party Group on Early Education and Childcare': 1705,
    'All-Party Group on Ethnic Minority Community': 1667,
    'All-Party Group on Food to Go': 2328,
    'All-Party Group on Football': 2220,
    'All-Party Group on Fuel Poverty': 2157,
    'All-Party Group on Further and Higher Education': 1742,
    'All-Party Group on Homelessness': 1895,
    'All-Party Group on Housing': 1282,
    'All-Party Group on International Development': 2195,
    'All-Party Group on Learning Disability': 1555,
    'All-Party Group on LGBTQIA+ Equality': 2156,
    'All-Party Group on Lung Health': 1720,
    'All-Party Group on Mental Health': 652,
    'All-Party Group on Micro and Small Business': 1781,
    'All-Party Group on MS and Neurology': 1968,
    'All-Party Group on Online Safety for Children and Young People': 2394,
    'All-Party Group on Parental Participation in Education': 1670,
    'All-Party Group on Policy and Public Data': 2274,
    'All-Party Group on Preventing Loneliness': 1629,
    'All-Party Group on Rare Disease': 2073,
    'All-Party Group on Reducing Harm Related to Gambling': 1666,
    'All-Party Group on Road Safety': 2248,
    'All-Party Group on Science, Technology, Engineering and Mathematics': 546,
    'All-Party Group on Skills': 2140,
    'All-Party Group on Social Enterprise': 1218,
    'All-Party group on Sports and Physical Recreation': 1465,  # small g
    'All-Party Group on Stroke': 2141,
    'All-Party Group on Suicide Prevention': 1507,
    'All-Party Group on Terminal Illness': 1719,
    'All-Party Group on Tourette\'s': 2184,
    'All-Party Group on UNSCR 1325, Women, Peace and Security': 283,
    'All-Party Group on Visual Impairment': 278,
    'All-Party Group on Voluntary and Community Sector': 2139,
    'All-Party Group on Women\'s Health': 1780,
    'All-Party Group on Youth Participation': 2138,
    #'All-Party Group on Cycling': 898,
    #'All-Party Group on Fairtrade': 1089,
    #'All-Party Group on Press Freedom and Media Sustainability': 1801,
    #'All-Party Group on Universal Basic Income': 2137,
}
apg_membership = []
for n in [n for n in current_apg_names if n in apg_links_dict.keys()]:
    cid = apg_links_dict[n]
    try:
        page = requests.get(f'https://aims.niassembly.gov.uk/mlas/apgdetails.aspx?&cid={cid}')
    except requests.exceptions.SSLError:
        print('SSLError for n={n}; wait and try again')
        time.sleep(5)
        try:
            page = requests.get(f'https://aims.niassembly.gov.uk/mlas/apgdetails.aspx?&cid={cid}')
        except requests.exceptions.SSLError:
            print('  another SSLError; skipping')
            continue

    soup = BeautifulSoup(page.text, features='lxml')
    apg_name = soup.find(name='h1').text

    apg_members = [el.text for el in soup.find(id='ctl00_MainContentPlaceHolder_CurrentMembershipPane_content').find_all(name='h4')]
    apg_members = [re.sub('Mr |Mrs |Ms |Miss |Dr |Sir | OBE| MBE| CBE| MC', '', m).strip() for m in apg_members]
    apg_members = [m for m in apg_members if m != '']

    apg_membership.append(pd.DataFrame({'apg_name': apg_name, 'member': apg_members}))

apg_membership = pd.concat(apg_membership)

# apg_membership.apg_name.nunique()  # 49 groups
# apg_membership.member.nunique()  # 75 members active
# apg_membership.member.value_counts().head(10)  # most active are on >20 groups!

print(f'\nFound APG membership for {apg_membership.apg_name.nunique()} groups and {apg_membership.member.nunique()} members')
apg_membership.to_csv(apg_membership_filepath, index=False)

# Note any new ones in current list that should have a link added above
new_apg_names = set(current_apg_names).difference(apg_membership.apg_name.tolist())
if len(new_apg_names) > 0:
    print('\n** New APGs found, not tracked: **', '\n- ' + '\n- '.join(new_apg_names))
