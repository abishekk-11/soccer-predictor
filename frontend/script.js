// Production serves the dashboard and API from one public address. Retain the
// local API address only when the static frontend is run on port 5500.
const API = window.location.hostname === "127.0.0.1" && window.location.port === "5500"
  ? "http://127.0.0.1:8000"
  : "";

const dom = {
  form: document.getElementById("predictionForm"),
  leagueSelect: document.getElementById("leagueSelect"),
  fixtureSelect: document.getElementById("fixtureSelect"),
  homeTeam: document.getElementById("homeTeam"),
  awayTeam: document.getElementById("awayTeam"),
  homePreview: document.getElementById("homeTeamPreview"),
  awayPreview: document.getElementById("awayTeamPreview"),
  predictButton: document.getElementById("predictButton"),
  appStatus: document.getElementById("appStatus"),
  formCard: document.getElementById("formCard"),
  formDescription: document.getElementById("formDescription"),
  homeFormPanel: document.getElementById("homeFormPanel"),
  awayFormPanel: document.getElementById("awayFormPanel"),
  playerCard: document.getElementById("playerCard"),
  playerSource: document.getElementById("playerSource"),
  homePlayerPanel: document.getElementById("homePlayerPanel"),
  awayPlayerPanel: document.getElementById("awayPlayerPanel"),
  resultPanel: document.getElementById("resultPanel"),
  resultTitle: document.getElementById("resultTitle"),
  resultDetail: document.getElementById("resultDetail"),
  confidenceRing: document.getElementById("confidenceRing"),
  confidenceValue: document.getElementById("confidenceValue"),
  probabilityLegend: document.getElementById("probabilityLegend"),
  heroMeta: document.getElementById("heroMeta"),
  hero: document.querySelector(".hero"),
  heroLeagueImage: document.getElementById("heroLeagueImage"),
  heroLeagueName: document.getElementById("heroLeagueName"),
  accuracyMetric: document.getElementById("accuracyMetric"),
  accuracyCopy: document.getElementById("accuracyCopy"),
  history: document.getElementById("history"),
  historyEmpty: document.getElementById("historyEmpty"),
  historyTableWrap: document.getElementById("historyTableWrap"),
  clearHistory: document.getElementById("clearHistory")
};

const clubProfiles = {
  "Arsenal": ["#e83b3b", "#dcb65d"], "Aston Villa": ["#76195e", "#7eb7d2"],
  "Barnsley": ["#d12638", "#ffffff"], "Birmingham": ["#2b84d1", "#ffffff"],
  "Blackburn": ["#2f5caa", "#d43f4f"], "Blackpool": ["#ed7b27", "#1c1c1c"],
  "Bolton": ["#294f83", "#ffffff"], "Bournemouth": ["#c9273d", "#161616"],
  "Bradford": ["#84283c", "#e0b83d"], "Brighton": ["#2374c7", "#ffffff"],
  "Burnley": ["#731d3d", "#78c4e5"], "Cardiff": ["#2365b2", "#ffffff"],
  "Charlton": ["#c92f3b", "#1a1a1a"], "Chelsea": ["#1c5fb6", "#ffffff"],
  "Coventry": ["#75bde8", "#142d52"], "Crystal Palace": ["#224e9b", "#d52a47"],
  "Derby": ["#222222", "#ffffff"], "Everton": ["#2455a6", "#ffffff"],
  "Fulham": ["#ffffff", "#d23c44"], "Huddersfield": ["#2672be", "#ffffff"],
  "Hull": ["#dfa425", "#1d1d1d"], "Ipswich": ["#1d4b9b", "#ffffff"],
  "Leeds": ["#f9f9f9", "#1a4ca0"], "Leicester": ["#2766b3", "#f5c748"],
  "Liverpool": ["#be2436", "#ffffff"], "Man City": ["#75bbe6", "#ffffff"],
  "Man United": ["#d83337", "#f1c548"], "Middlesboro": ["#d22f3d", "#ffffff"],
  "Middlesbrough": ["#d22f3d", "#ffffff"], "Newcastle": ["#202020", "#f0f0f0"],
  "Norwich": ["#e4c339", "#237843"], "Nott'm Forest": ["#d7343f", "#ffffff"],
  "Oldham": ["#2668b0", "#ffffff"], "Portsmouth": ["#2956a0", "#ffffff"],
  "QPR": ["#3179c5", "#ffffff"], "Reading": ["#2d62ae", "#ffffff"],
  "Sheffield United": ["#d83241", "#ffffff"], "Sheffield Weds": ["#2c67b0", "#ffffff"],
  "Southampton": ["#d93643", "#ffffff"], "Stoke": ["#d63a43", "#ffffff"],
  "Sunderland": ["#d63143", "#ffffff"], "Swansea": ["#202020", "#ffffff"],
  "Swindon": ["#c92f3c", "#ffffff"], "Tottenham": ["#f4f6f7", "#1b3767"],
  "Watford": ["#e2b222", "#1b1b1b"], "West Brom": ["#2856a5", "#ffffff"],
  "West Ham": ["#7a2744", "#6ab9df"], "Wigan": ["#2b66b4", "#ffffff"],
  "Wimbledon": ["#255baf", "#f2c837"], "Wolves": ["#e7a922", "#1a1a1a"]
};

const badgeAliases = {
  "Man City": "Manchester City", "Man United": "Manchester United", "Middlesboro": "Middlesbrough",
  "Nott'm Forest": "Nottingham Forest", "QPR": "Queens Park Rangers", "Sheffield Weds": "Sheffield Wednesday",
  "West Brom": "West Bromwich Albion", "Wolves": "Wolverhampton Wanderers"
};

// PSG's current circular Paris/Eiffel Tower crest is pinned to a stable team
// image rather than depending on a third-party name search.
const officialBadgeUrls = {
  "Paris Saint-Germain": "https://a.espncdn.com/i/teamlogos/soccer/500/160.png"
};

const leagueHeroArt = {
  premier_league: { image: "assets/leagues/premier-league.png", alt: "Premier League trophy artwork" },
  la_liga: { image: "assets/leagues/la-liga.png", alt: "La Liga trophy artwork" },
  bundesliga: { image: "assets/leagues/bundesliga.png", alt: "Bundesliga trophy artwork" },
  ligue_1: { image: "assets/leagues/ligue-1.png", alt: "Ligue 1 trophy artwork" },
  serie_a: { image: "assets/leagues/serie-a.png", alt: "Serie A trophy artwork" }
};

let history = safeHistory();
let probabilityChart;
let accuracyGraph;
let validationAccuracy = 0;
let matchupVersion = 0;
let currentLeague = "premier_league";
let currentLeagueName = "Premier League";
let currentFixtures = [];
let selectedFixture = null;
let changingFixture = false;
const badgeRequests = new Map();

function safeHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem("pitchIqHistory"));
    return Array.isArray(saved) ? saved.slice(0, 12) : [];
  } catch {
    return [];
  }
}

function setStatus(message = "", type = "") {
  dom.appStatus.textContent = message;
  dom.appStatus.style.color = type === "success" ? "var(--lime)" : "";
}

function updateHeroLeague(leagueKey, leagueName = "Premier League") {
  const art = leagueHeroArt[leagueKey] || leagueHeroArt.premier_league;
  dom.hero.dataset.league = leagueKey || "premier_league";
  dom.heroLeagueName.textContent = leagueName;
  if (dom.heroLeagueImage.src.endsWith(art.image)) return;
  dom.heroLeagueImage.classList.add("is-switching");
  const finishHeroSwap = () => dom.heroLeagueImage.classList.remove("is-switching");
  dom.heroLeagueImage.addEventListener("load", finishHeroSwap, { once: true });
  dom.heroLeagueImage.addEventListener("error", finishHeroSwap, { once: true });
  dom.heroLeagueImage.src = art.image;
  dom.heroLeagueImage.alt = art.alt;
}

function initials(team) {
  const known = { "Man City": "MC", "Man United": "MU", "Nott'm Forest": "NF", "QPR": "QPR", "West Brom": "WB", "Sheffield United": "SU", "Sheffield Weds": "SW" };
  if (known[team]) return known[team];
  return team.split(/\s+/).map(word => word[0]).join("").slice(0, 3).toUpperCase();
}

function crestFor(team, size = "") {
  const crest = document.createElement("span");
  crest.className = `team-crest ${size}`.trim();
  crest.style.setProperty("--crest", `linear-gradient(135deg, ${(clubProfiles[team] || ["#287d6c", "#b1d350"])[0]} 0 56%, ${(clubProfiles[team] || ["#287d6c", "#b1d350"])[1]} 56% 100%)`);
  crest.textContent = initials(team);
  crest.setAttribute("aria-label", `${team} club crest`);
  return crest;
}

function teamPreview(team, venue) {
  const fragment = document.createDocumentFragment();
  fragment.append(crestFor(team));
  const copy = document.createElement("div");
  copy.className = "team-preview-copy";
  const name = document.createElement("strong");
  name.textContent = team;
  const label = document.createElement("span");
  label.textContent = venue;
  copy.append(name, label);
  fragment.append(copy);
  return fragment;
}

async function applyOfficialBadge(crest, team) {
  const pinnedBadge = officialBadgeUrls[team];
  if (pinnedBadge) {
    const image = new Image();
    image.alt = "";
    image.src = pinnedBadge;
    image.addEventListener("error", () => image.remove(), { once: true });
    crest.replaceChildren(image);
    return;
  }
  const searchName = badgeAliases[team] || team;
  if (!badgeRequests.has(searchName)) {
    badgeRequests.set(searchName, fetch(`https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t=${encodeURIComponent(searchName)}`)
      .then(response => response.ok ? response.json() : null)
      .then(data => data?.teams?.[0]?.strBadge || null)
      .catch(() => null));
  }
  const badgeUrl = await badgeRequests.get(searchName);
  if (!badgeUrl || !crest.isConnected) return;
  const image = new Image();
  image.alt = "";
  image.src = badgeUrl;
  image.addEventListener("error", () => image.remove(), { once: true });
  crest.replaceChildren(image);
}

function renderTeamPreview(target, team, venue) {
  target.replaceChildren(teamPreview(team, venue));
  const crest = target.querySelector(".team-crest");
  applyOfficialBadge(crest, team);
}

function formMatchRow(match) {
  const row = document.createElement("li");
  row.className = "form-fixture";

  const date = document.createElement("time");
  date.dateTime = match.date || "";
  date.textContent = formatDate(match.date);

  const opponent = document.createElement("span");
  opponent.className = "form-opponent";
  opponent.textContent = `${match.venue === "Home" ? "vs" : "@"} ${match.opponent || "—"}`;

  const score = document.createElement("strong");
  score.className = "form-score";
  score.textContent = `${match.goals_for ?? "—"}–${match.goals_against ?? "—"}`;

  const result = document.createElement("span");
  result.className = `form-fixture-result ${match.result || ""}`;
  result.textContent = match.result || "—";
  row.append(date, opponent, score, result);
  return row;
}

function createFormTeam(team, data) {
  const panel = document.createElement("div");
  const header = document.createElement("div");
  header.className = "form-team-header";
  const crest = crestFor(team);
  const copy = document.createElement("div");
  const title = document.createElement("h3");
  title.textContent = team;
  const subtitle = document.createElement("span");
  subtitle.className = "form-team-subtitle";
  subtitle.textContent = `${data.source_competition || "League"} · ${data.source_season || "2025/26"}`;
  copy.append(title, subtitle);
  header.append(crest, copy);

  const pips = document.createElement("div");
  pips.className = "form-pips";
  const form = Array.isArray(data.form) && data.form.length ? data.form : ["—"];
  form.forEach(result => {
    const pip = document.createElement("span");
    pip.className = `form-pip ${result}`;
    pip.textContent = result;
    pip.title = result === "W" ? "Win" : result === "D" ? "Draw" : result === "L" ? "Loss" : "No result";
    pips.append(pip);
  });

  const stats = document.createElement("div");
  stats.className = "form-stats";
  [["Goals", data.goals_scored], ["Conceded", data.goals_conceded]].forEach(([label, value]) => {
    const stat = document.createElement("span");
    stat.className = "form-stat";
    const number = document.createElement("b");
    number.textContent = value ?? 0;
    stat.append(number, label);
    stats.append(stat);
  });
  const fixtures = document.createElement("ol");
  fixtures.className = "form-fixtures";
  (Array.isArray(data.matches) ? data.matches : []).forEach(match => fixtures.append(formMatchRow(match)));

  const source = document.createElement("a");
  source.className = "form-source";
  source.href = data.source_url || "#";
  source.target = "_blank";
  source.rel = "noreferrer";
  source.textContent = `${data.live ? "Live results" : "Verified results"} · through ${formatDate(data.data_through)}`;

  panel.append(header, pips, stats, fixtures, source);
  applyOfficialBadge(crest, team);
  return panel;
}

function playerInitials(name) {
  return String(name || "?").split(/\s+/).map(part => part[0]).join("").slice(0, 2).toUpperCase();
}

function playerPhoto(player, size = "") {
  const photo = document.createElement("span");
  photo.className = `player-photo ${size}`.trim();
  const fallback = document.createElement("span");
  fallback.textContent = playerInitials(player.name);
  photo.append(fallback);
  if (!player.photo_url) return photo;
  const image = new Image();
  image.alt = `${player.name} headshot`;
  image.src = player.photo_url;
  image.addEventListener("load", () => photo.replaceChildren(image), { once: true });
  image.addEventListener("error", () => image.remove(), { once: true });
  return photo;
}

function statValue(value, decimals = 0) {
  if (value === null || value === undefined || value === "") return "—";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  return decimals ? numeric.toFixed(decimals) : Math.round(numeric).toLocaleString();
}

function targetCard(player, label) {
  const card = document.createElement("div");
  card.className = "player-target";
  card.append(playerPhoto(player, "target-photo"));
  const copy = document.createElement("div");
  copy.className = "player-target-copy";
  const name = document.createElement("strong");
  name.textContent = player.name;
  const position = document.createElement("span");
  position.textContent = label ? `${label} · ${player.position}` : player.position;
  copy.append(name, position);
  const scores = document.createElement("div");
  scores.className = "player-target-scores";
  [["Goal", player.score_likelihood], ["Assist", player.assist_likelihood]].forEach(([label, value]) => {
    const score = document.createElement("span");
    const valueElement = document.createElement("b");
    valueElement.textContent = percent(value);
    score.append(valueElement, ` ${label}`);
    scores.append(score);
  });
  card.append(copy, scores);
  return card;
}

function playerStatRow(player, statBasis) {
  const row = document.createElement("div");
  row.className = "player-stat-row";
  const identity = document.createElement("div");
  identity.className = "player-identity";
  identity.append(playerPhoto(player));
  const name = document.createElement("div");
  const label = document.createElement("strong");
  label.textContent = player.name;
  const position = document.createElement("span");
  position.textContent = player.position;
  name.append(label, position);
  identity.append(name);

  const latest = document.createElement("div");
  latest.className = "player-latest-stats";
  const participationValue = statBasis === "appearances" ? player.latest_appearances : player.latest_minutes;
  const participationLabel = statBasis === "appearances" ? "Apps" : "Min";
  [["G", player.latest_goals], ["A", player.latest_assists], [participationLabel, participationValue], ["xG", player.latest_xg], ["xA", player.latest_xa]].forEach(([labelText, value]) => {
    const stat = document.createElement("span");
    const label = document.createElement("small");
    label.textContent = labelText;
    const number = document.createElement("b");
    number.textContent = labelText === "xG" || labelText === "xA" ? statValue(value, 1) : statValue(value);
    stat.append(label, number);
    latest.append(stat);
  });

  const career = document.createElement("div");
  career.className = "player-career-stats";
  const contribution = document.createElement("b");
  contribution.textContent = player.has_history ? `${player.career_goals}G + ${player.career_assists}A` : "—";
  const seasons = document.createElement("span");
  const careerParticipation = statBasis === "appearances"
    ? `${statValue(player.career_appearances)} apps`
    : `${statValue(player.career_minutes)} min`;
  seasons.textContent = player.has_history ? `${player.seasons_played} seasons · ${careerParticipation}` : "No completed record";
  career.append(contribution, seasons);

  row.append(identity, latest, career);
  return row;
}

function createPlayerTeam(teamData, venue, playerData) {
  const panel = document.createElement("div");
  const header = document.createElement("div");
  header.className = "player-team-header";
  const crest = crestFor(teamData.team);
  const copy = document.createElement("div");
  const title = document.createElement("h3");
  title.textContent = teamData.team;
  const summary = document.createElement("span");
  summary.className = "player-team-summary";
  summary.textContent = `${venue} · ${teamData.players.length} available squad players · ${teamData.projected_team_goals.toFixed(2)} projected goals`;
  copy.append(title, summary);
  header.append(crest, copy);

  const targetTitle = document.createElement("p");
  targetTitle.className = "player-target-title";
  targetTitle.textContent = "Fixture scorer & assist predictions";
  const targetGrid = document.createElement("div");
  targetGrid.className = "player-target-grid";
  const eligiblePlayers = teamData.players.filter(player => (
    player.has_history && (player.score_likelihood > 0 || player.assist_likelihood > 0)
  ));
  const fixturePicks = playerData.fixture_predictions?.[venue.toLowerCase()] || {};
  const playerForPick = pick => teamData.players.find(player => player.player_code === pick?.player_code);
  const scoreLeader = playerForPick(fixturePicks.likely_scorer)
    || [...eligiblePlayers].sort((a, b) => b.score_likelihood - a.score_likelihood)[0];
  const assistLeader = playerForPick(fixturePicks.likely_assister)
    || [...eligiblePlayers].sort((a, b) => b.assist_likelihood - a.assist_likelihood)[0];
  const targets = [];
  if (scoreLeader && assistLeader?.player_code === scoreLeader.player_code) {
    targets.push([scoreLeader, "Likely scorer & assister"]);
  } else {
    if (scoreLeader) targets.push([scoreLeader, "Likely scorer"]);
    if (assistLeader) targets.push([assistLeader, "Likely assister"]);
  }
  eligiblePlayers.some(player => {
    if (targets.some(([target]) => target.player_code === player.player_code)) return false;
    targets.push([player, "Next best signal"]);
    return true;
  });
  if (targets.length) {
    targets.forEach(([player, label]) => targetGrid.append(targetCard(player, label)));
  } else {
    const empty = document.createElement("p");
    empty.className = "player-target-empty";
    empty.textContent = "No usable completed-player sample is linked to this roster yet.";
    targetGrid.append(empty);
  }

  const allStatsTitle = document.createElement("p");
  allStatsTitle.className = "player-all-title";
  allStatsTitle.textContent = "All player stats";
  const columnNote = document.createElement("span");
  columnNote.className = "player-column-note";
  const participationLabel = playerData.player_stat_basis === "appearances" ? "appearances" : "minutes";
  columnNote.textContent = `2025/26: G, A, ${participationLabel}, xG, xA · Ten-season portfolio: ${playerData.historical_window}`;
  const table = document.createElement("div");
  table.className = "player-stat-list";
  teamData.players.forEach(player => table.append(playerStatRow(player, playerData.player_stat_basis)));

  panel.append(header, targetTitle, targetGrid, allStatsTitle, columnNote, table);
  return panel;
}

function renderPlayerMatchup(data) {
  dom.homePlayerPanel.className = "player-team-panel";
  dom.awayPlayerPanel.className = "player-team-panel away";
  dom.homePlayerPanel.replaceChildren(createPlayerTeam(data.home, "Home", data));
  dom.awayPlayerPanel.replaceChildren(createPlayerTeam(data.away, "Away", data));
  const squadRefresh = data.squad_refresh || {};
  const squadsLive = Boolean(squadRefresh.home?.live && squadRefresh.away?.live);
  const refreshMinutes = Number(squadRefresh.refresh_minutes || 60);
  const squadLabel = squadsLive
    ? `live squads checked automatically every ${refreshMinutes} minutes`
    : "latest cached squad shown while the live roster feed refreshes";
  dom.playerSource.textContent = `${data.historical_window} completed data · ${squadLabel}`;
  dom.playerCard.classList.remove("hidden");
  applyOfficialBadge(dom.homePlayerPanel.querySelector(".team-crest"), data.home.team);
  applyOfficialBadge(dom.awayPlayerPanel.querySelector(".team-crest"), data.away.team);
}

function leagueQuery(path) {
  return `${API}${path}${path.includes("?") ? "&" : "?"}league=${encodeURIComponent(currentLeague)}`;
}

function fixtureLabel(fixture) {
  return `${formatDate(fixture.date)} · ${fixture.home_team} vs ${fixture.away_team}`;
}

function populateTeams(teams) {
  [dom.homeTeam, dom.awayTeam].forEach(select => {
    select.replaceChildren();
    teams.forEach(team => {
      const option = document.createElement("option");
      option.value = team;
      option.textContent = team;
      select.append(option);
    });
    select.disabled = false;
  });
}

function selectFixture(index, refresh = true) {
  const fixture = currentFixtures[index];
  if (!fixture) return;
  changingFixture = true;
  selectedFixture = fixture;
  dom.fixtureSelect.value = String(index);
  dom.homeTeam.value = fixture.home_team;
  dom.awayTeam.value = fixture.away_team;
  changingFixture = false;
  if (refresh) updateMatchup();
}

function syncFixtureToTeams() {
  const fixtureIndex = currentFixtures.findIndex(fixture => (
    fixture.home_team === dom.homeTeam.value && fixture.away_team === dom.awayTeam.value
  ));
  if (fixtureIndex >= 0) selectFixture(fixtureIndex, false);
}

async function loadLeague() {
  setStatus(`Loading ${currentLeagueName} fixtures…`);
  dom.predictButton.disabled = true;
  dom.fixtureSelect.disabled = true;
  try {
    // Let the first request prepare a league once, then fetch the remaining
    // lightweight views in parallel from that cached runtime.
    const teamsResponse = await fetch(leagueQuery("/teams"));
    if (!teamsResponse.ok) throw new Error("League data is unavailable.");
    const teamsData = await teamsResponse.json();
    const [fixturesResponse, infoResponse] = await Promise.all([
      fetch(leagueQuery("/fixtures")),
      fetch(leagueQuery("/model-info"))
    ]);
    if (!fixturesResponse.ok) throw new Error("League data is unavailable.");
    const fixturesData = await fixturesResponse.json();
    const teams = Array.isArray(teamsData.teams) ? teamsData.teams : [];
    currentFixtures = Array.isArray(fixturesData.fixtures) ? fixturesData.fixtures : [];
    if (teams.length < 2 || !currentFixtures.length) throw new Error("There are not enough official fixtures to build a match.");
    currentLeagueName = fixturesData.league_name || teamsData.league_name || currentLeagueName;
    populateTeams(teams);
    dom.fixtureSelect.replaceChildren();
    currentFixtures.forEach((fixture, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = fixtureLabel(fixture);
      dom.fixtureSelect.append(option);
    });
    dom.fixtureSelect.disabled = false;
    selectFixture(0, false);
    if (infoResponse.ok) applyModelInfo(await infoResponse.json());
    dom.predictButton.disabled = false;
    setStatus("");
    await updateMatchup();
  } catch (error) {
    dom.formCard.classList.add("hidden");
    dom.playerCard.classList.add("hidden");
    setStatus("Couldn’t load this league. Start the local API and refresh.");
    console.error(error);
  }
}

async function loadLeagues() {
  setStatus("Loading supported leagues…");
  const wakeUpNotice = window.setTimeout(() => {
    setStatus("Waking up the predictor… free hosting can take up to a minute after inactivity.");
  }, 2500);
  try {
    const response = await fetch(`${API}/leagues`);
    if (!response.ok) throw new Error("League list unavailable.");
    const data = await response.json();
    const leagues = Array.isArray(data.leagues) ? data.leagues : [];
    if (!leagues.length) throw new Error("No leagues available.");
    dom.leagueSelect.replaceChildren();
    leagues.forEach(league => {
      const option = document.createElement("option");
      option.value = league.key;
      option.textContent = league.name;
      dom.leagueSelect.append(option);
    });
    currentLeague = leagues.some(league => league.key === currentLeague) ? currentLeague : leagues[0].key;
    currentLeagueName = leagues.find(league => league.key === currentLeague)?.name || leagues[0].name;
    dom.leagueSelect.value = currentLeague;
    dom.leagueSelect.disabled = false;
    updateHeroLeague(currentLeague, currentLeagueName);
    await loadLeague();
  } catch (error) {
    setStatus("Couldn’t connect to the predictor. Start the local API and refresh.");
    console.error(error);
  } finally {
    window.clearTimeout(wakeUpNotice);
  }
}

function applyModelInfo(info) {
  currentLeagueName = info?.league_name || currentLeagueName;
  updateHeroLeague(currentLeague, currentLeagueName);
  const accuracy = Number(info?.validation?.accuracy || 0) * 100;
  validationAccuracy = accuracy;
  dom.accuracyMetric.textContent = accuracy ? `${accuracy.toFixed(1)}%` : "—";
  dom.accuracyCopy.textContent = info?.validation_method || "Chronological validation details are unavailable.";
  if (info?.team_count && info?.fixtures_loaded) {
    dom.heroMeta.replaceChildren();
    [[info.team_count, `clubs in ${currentLeagueName}`], [info.fixtures_loaded, "official fixtures loaded"]].forEach(([value, label]) => {
      const item = document.createElement("span");
      const number = document.createElement("b");
      number.textContent = value;
      item.append(number, ` ${label}`);
      dom.heroMeta.append(item);
    });
  }
  accuracyChart(validationAccuracy);
}

async function updateMatchup() {
  const home = dom.homeTeam.value;
  const away = dom.awayTeam.value;
  const fixtureDate = selectedFixture?.date;
  const requestVersion = ++matchupVersion;
  if (!home || !away) return;
  renderTeamPreview(dom.homePreview, home, "Home side");
  renderTeamPreview(dom.awayPreview, away, "Away side");
  if (home === away) {
    dom.formCard.classList.add("hidden");
    dom.playerCard.classList.add("hidden");
    setStatus("Choose two different teams to generate a forecast.");
    return;
  }
  setStatus("");
  // Player portfolios are richer (and therefore slower) than the form cards.
  // They load after a user asks for a forecast so changing fixtures remains
  // responsive and the main prediction never waits on roster calculations.
  dom.playerCard.classList.add("hidden");
  const formRequest = Promise.all([
      fetch(leagueQuery(`/form/${encodeURIComponent(home)}`)),
      fetch(leagueQuery(`/form/${encodeURIComponent(away)}`))
  ]).then(async ([homeResponse, awayResponse]) => {
    if (!homeResponse.ok || !awayResponse.ok) throw new Error("Form data unavailable");
    return Promise.all([homeResponse.json(), awayResponse.json()]);
  });
  const formResult = await Promise.allSettled([formRequest]);
  if (requestVersion !== matchupVersion || home !== dom.homeTeam.value || away !== dom.awayTeam.value) return;

  if (formResult[0].status === "fulfilled") {
    const [homeData, awayData] = formResult[0].value;
    const hasLiveForm = Boolean(homeData.live || awayData.live);
    const currentSeason = String(homeData.current_season || awayData.current_season || "2026-27").replace(/^(\d{4})-(\d{2}).*$/, "$1/$2");
    const refreshMinutes = Number(homeData.refresh_minutes || awayData.refresh_minutes || 15);
    dom.formDescription.textContent = hasLiveForm
      ? `Latest completed ${currentSeason} league fixtures, refreshed automatically every ${refreshMinutes} minutes. Shown oldest to newest.`
      : `No completed ${currentSeason} league fixtures yet. Showing the latest available form, oldest to newest.`;
    dom.homeFormPanel.className = "form-team";
    dom.awayFormPanel.className = "form-team away";
    dom.homeFormPanel.replaceChildren(createFormTeam(home, homeData));
    dom.awayFormPanel.replaceChildren(createFormTeam(away, awayData));
    dom.formCard.classList.remove("hidden");
  } else {
    dom.formCard.classList.add("hidden");
    setStatus("Recent form is temporarily unavailable.");
    console.error(formResult[0].reason);
  }
}

async function loadPlayerMatchup(home, away, fixtureDate, requestVersion) {
  try {
    const response = await fetch(leagueQuery(
      `/match-players?home_team=${encodeURIComponent(home)}&away_team=${encodeURIComponent(away)}&fixture_date=${encodeURIComponent(fixtureDate || "")}`
    ));
    if (!response.ok) throw new Error("Player data unavailable");
    const playerData = await response.json();
    if (requestVersion !== matchupVersion || home !== dom.homeTeam.value || away !== dom.awayTeam.value) return;
    renderPlayerMatchup(playerData);
  } catch (error) {
    dom.playerCard.classList.add("hidden");
    console.error(error);
  }
}

function percent(value) { return `${Math.round(Number(value || 0) * 100)}%`; }

function drawProbability(probabilities) {
  const values = [probabilities.home || 0, probabilities.draw || 0, probabilities.away || 0].map(value => Math.round(value * 100));
  if (probabilityChart) probabilityChart.destroy();
  probabilityChart = new Chart(document.getElementById("probabilityChart"), {
    type: "doughnut",
    data: { labels: ["Home win", "Draw", "Away win"], datasets: [{ data: values, backgroundColor: ["#d7f767", "#f2c86f", "#5ae1cb"], borderWidth: 0, hoverOffset: 5, borderRadius: 4 }] },
    options: { responsive: true, maintainAspectRatio: false, cutout: "73%", plugins: { legend: { display: false }, tooltip: { backgroundColor: "#10282b", padding: 10, displayColors: false, callbacks: { label: context => `${context.label}: ${context.parsed}%` } } } }
  });
  const labels = ["Home win", "Draw", "Away win"];
  dom.probabilityLegend.replaceChildren(...labels.map((label, index) => {
    const line = document.createElement("p");
    const dot = document.createElement("span");
    dot.className = `legend-dot ${["home-dot", "draw-dot", "away-dot"][index]}`;
    const text = document.createElement("span");
    text.textContent = label;
    const value = document.createElement("strong");
    value.textContent = `${values[index]}%`;
    line.append(dot, text, value);
    return line;
  }));
}

function renderResult(data) {
  const homeWins = data.prediction === "Home Team Wins";
  const awayWins = data.prediction === "Away Team Wins";
  dom.resultTitle.textContent = homeWins ? data.home_team : awayWins ? data.away_team : "Draw projected";
  const methodNote = `Uses pre-match team ratings, home/away form and recent league results through ${data.data_through || "the latest completed season"}.`;
  dom.resultDetail.textContent = homeWins ? `${data.home_team} are favoured on home turf. ${methodNote}` : awayWins ? `${data.away_team} are favoured to take the points. ${methodNote}` : `The model expects a tightly balanced contest. ${methodNote}`;
  const confidence = percent(data.confidence);
  dom.confidenceValue.textContent = confidence;
  dom.confidenceRing.style.setProperty("--confidence", confidence);
  dom.confidenceRing.setAttribute("aria-label", `${confidence} prediction confidence`);
  dom.resultPanel.classList.remove("hidden");
}

async function predictMatch(event) {
  event.preventDefault();
  const home = dom.homeTeam.value;
  const away = dom.awayTeam.value;
  if (home === away) { setStatus("Choose two different teams to generate a forecast."); return; }
  dom.predictButton.disabled = true;
  dom.predictButton.querySelector("span:nth-child(2)").textContent = "Analysing matchup…";
  setStatus("");
  try {
    const response = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        league: currentLeague,
        home_team: home,
        away_team: away,
        fixture_date: selectedFixture?.date || null
      })
    });
    if (!response.ok) throw new Error("Prediction request failed");
    const data = await response.json();
    data.createdAt = new Date().toISOString();
    renderResult(data);
    drawProbability(data.probabilities);
    history = [data, ...history].slice(0, 12);
    localStorage.setItem("pitchIqHistory", JSON.stringify(history));
    renderHistory();
    // Do not delay the forecast while detailed scorer and assist portfolios
    // are prepared. They appear immediately afterward for the same fixture.
    void loadPlayerMatchup(home, away, selectedFixture?.date, matchupVersion);
  } catch (error) {
    setStatus("The forecast could not be generated. Please try again.");
    console.error(error);
  } finally {
    dom.predictButton.disabled = false;
    dom.predictButton.querySelector("span:nth-child(2)").textContent = "Generate prediction";
  }
}

function historyOutcome(entry) {
  if (entry.prediction === "Home Team Wins") return `${entry.home_team} win`;
  if (entry.prediction === "Away Team Wins") return `${entry.away_team} win`;
  return "Draw";
}

function formatDate(value) {
  if (!value) return "Earlier";
  // API fixture dates are calendar dates, not instants.  Use midday local time
  // so viewers west of UTC do not see the previous day.
  const date = new Date(/^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T12:00:00` : value);
  return Number.isNaN(date.getTime()) ? "Earlier" : date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function renderHistory() {
  const hasHistory = history.length > 0;
  dom.historyEmpty.classList.toggle("hidden", hasHistory);
  dom.historyTableWrap.classList.toggle("hidden", !hasHistory);
  dom.clearHistory.classList.toggle("hidden", !hasHistory);
  dom.history.replaceChildren();
  history.forEach(entry => {
    const row = document.createElement("tr");
    const match = document.createElement("td");
    const matchup = document.createElement("div");
    matchup.className = "history-match";
    const league = document.createElement("span");
    league.textContent = entry.league_name || "League";
    matchup.append(league, document.createTextNode(entry.home_team || "Home"));
    const separator = document.createElement("span");
    separator.textContent = "vs";
    matchup.append(separator, document.createTextNode(entry.away_team || "Away"));
    match.append(matchup);
    const outcome = document.createElement("td");
    const outcomeTag = document.createElement("span");
    outcomeTag.className = "history-outcome";
    outcomeTag.textContent = historyOutcome(entry);
    outcome.append(outcomeTag);
    const confidence = document.createElement("td");
    confidence.className = "history-confidence";
    confidence.textContent = percent(entry.confidence);
    const date = document.createElement("td");
    date.textContent = formatDate(entry.createdAt);
    row.append(match, outcome, confidence, date);
    dom.history.append(row);
  });
}

function accuracyChart(value = validationAccuracy) {
  if (accuracyGraph) accuracyGraph.destroy();
  accuracyGraph = new Chart(document.getElementById("accuracyChart"), {
    type: "bar",
    data: { labels: ["Model accuracy"], datasets: [{ data: [value], backgroundColor: "#d7f767", borderRadius: 8, borderSkipped: false, barThickness: 28 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { display: false, grid: { display: false } }, y: { beginAtZero: true, max: 100, ticks: { color: "#72877c", font: { family: "DM Mono", size: 9 }, callback: value => `${value}%`, maxTicksLimit: 4 }, grid: { color: "rgba(203, 231, 216, .10)", drawBorder: false } } },
      plugins: { legend: { display: false }, tooltip: { backgroundColor: "#10282b", displayColors: false, callbacks: { label: context => `${context.parsed.y}% accuracy` } } }
    }
  });
}

dom.leagueSelect.addEventListener("change", async () => {
  currentLeague = dom.leagueSelect.value;
  currentLeagueName = dom.leagueSelect.options[dom.leagueSelect.selectedIndex]?.textContent || currentLeagueName;
  updateHeroLeague(currentLeague, currentLeagueName);
  await loadLeague();
});
dom.fixtureSelect.addEventListener("change", () => selectFixture(Number(dom.fixtureSelect.value)));
dom.homeTeam.addEventListener("change", () => {
  if (changingFixture) return;
  syncFixtureToTeams();
  updateMatchup();
});
dom.awayTeam.addEventListener("change", () => {
  if (changingFixture) return;
  syncFixtureToTeams();
  updateMatchup();
});
dom.form.addEventListener("submit", predictMatch);
dom.clearHistory.addEventListener("click", () => {
  history = [];
  localStorage.removeItem("pitchIqHistory");
  renderHistory();
});

renderHistory();
accuracyChart();
loadLeagues();
