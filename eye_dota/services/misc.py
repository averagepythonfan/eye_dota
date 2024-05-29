query = """
WITH
    players_table (match_id, players) AS (
        SELECT
            matches.match_id,
            ARRAY_AGG(player_matches.account_id 
        ORDER BY
            player_matches.player_slot) AS players_id
        FROM
            matches     
        JOIN
            player_matches using(match_id)     
        JOIN
            leagues 
                ON matches.leagueid = leagues.leagueid          
        WHERE
            matches.start_time > extract(epoch from timestamp '{start_date}')
            AND matches.start_time < extract(epoch from timestamp '{end_date}')
            AND leagues.tier IN ('professional', 'premium')     
        GROUP BY
            matches.match_id),
    
    picks_table (match_id, picks) AS (
        SELECT
            matches.match_id,
            ARRAY_AGG(picks_bans.hero_id 
        ORDER BY
            picks_bans.team ASC) picks          
        FROM
            matches     
        JOIN
            picks_bans using(match_id)     
        JOIN
            leagues 
                ON matches.leagueid = leagues.leagueid          
        WHERE
            matches.start_time > extract(epoch from timestamp '{start_date}')
            AND matches.start_time < extract(epoch from timestamp '{end_date}')    
            AND picks_bans.is_pick = true     
            AND leagues.tier IN ('professional', 'premium')     
        GROUP BY
            matches.match_id),

    match_data_table (match_id,
                        start_match,
                        league_id,
                        tier,
                        radiant_team_id,
                        dire_team_id,
                        radiant_win,
                        rscore,
                        dscore,
                        durations) AS (
        SELECT
            matches.match_id,
            matches.start_time::BIGINT * 1000 AS start_match,
            matches.leagueid AS league_id,
            CASE WHEN leagues.tier = 'premium' THEN true ELSE false END AS tier,
            matches.radiant_team_id,
            matches.dire_team_id,
            matches.radiant_win,
            matches.radiant_score radiant_score,
            matches.dire_score dire_score,
            matches.duration:: int / 60 as duration,
            match_patch.patch as patch,
            matches.radiant_gold_adv as gold,
            matches.radiant_xp_adv as xp
        FROM matches
        JOIN leagues ON matches.leagueid = leagues.leagueid
        RIGHT JOIN match_patch on matches.match_id = match_patch.match_id
        
        WHERE leagues.tier IN ('professional', 'premium')
        AND matches.start_time > extract(epoch from timestamp '{start_date}')
        AND matches.start_time < extract(epoch from timestamp '{end_date}')
        AND matches.radiant_team_id IS NOT NULL
        AND matches.dire_team_id IS NOT NULL
        AND matches.radiant_gold_adv IS NOT NULL
        AND matches.radiant_xp_adv IS NOT NULL
        
        GROUP BY matches.match_id, leagues.tier, match_patch.patch
        ORDER BY matches.match_id ASC)
SELECT
    * 
FROM
    match_data_table 
INNER JOIN
    picks_table using(match_id)
INNER JOIN
    players_table using(match_id)
LIMIT {limit}"""