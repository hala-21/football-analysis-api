import json
from typing import Dict, List, Optional

def load_teams_data() -> List[Dict]:
    with open('data/teams.json', 'r') as file:
        return json.load(file)

def load_players_data(team_name: Optional[str] = None) -> List[Dict]:
    with open('data/players.json', 'r') as file:
        players = json.load(file)
    
    if team_name:
        return [player for player in players if player['team_name'].lower() == team_name.lower()]
    return players