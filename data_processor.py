from typing import Dict, List, Optional
import data_loader

def process_player_stats(player_name: str) -> Optional[Dict]:
    players = data_loader.load_players_data()
    for player in players:
        if player['name'].lower() == player_name.lower():
            return {
                'name': player['name'],
                'goals': player['goals'],
                'assists': player['assists'],
                'yellow_cards': player['yellow_cards'],
                'red_cards': player['red_cards']
            }
    return None

def process_team_stats(team_name: str) -> Optional[Dict]:
    players = data_loader.load_players_data(team_name)
    if not players:
        return None
    
    total_goals = sum(player['goals'] for player in players)
    total_assists = sum(player['assists'] for player in players)
    avg_goals = total_goals / len(players)
    avg_assists = total_assists / len(players)
    
    return {
        'team_name': team_name,
        'total_goals': total_goals,
        'total_assists': total_assists,
        'average_goals': avg_goals,
        'average_assists': avg_assists
    }