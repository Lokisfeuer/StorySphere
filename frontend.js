function IDexists(id) {
    const element = document.getElementById(id);
    // If it isn't "undefined" and it isn't "null", then it exists.
    return typeof (element) != 'undefined' && element != null;
}
function getall(otype, aspect='name') {
    var id = "string";
    var names = [];
    for (let i = 0; i < 100; i++) {
        id = otype + '_' + i.toString() + '_' + aspect;
        if(IDexists(id)){
            names.push(document.getElementById(id).value);
        } else{
            return names;
        }
    }
}

function dropdown(ls) {
    var com = "        <option value=''></option>\n";
    for (let i = 0; i < ls.length; i++) {
        var name = ls[i];
        var name_com = "        <option value='" + name + "'>" + name + "</option>\n";
        com = com + name_com;
    }
    return com;
}

function newcell(row, modus, id, dropdown_ls = [], multiple = '') {
    // if its dropdown with multiple just add multiple at the end of the id
    const modi_sta = ["<input type='text' id=", "<textarea id=", "<select id="];
    const modi_end = [">", "></textarea>", ">\n" + dropdown_ls + "</select>"];
    // modus is 0;1 or 2
    if(multiple!='') {
        const x = "'";
    } else {
        const x = "";
    }
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td>" + modi_sta[modus] + id + x + " name=" + id + multiple + modi_end[modus] + "</td></tr>";
}

function newadv() {
    document.getElementById('myvariables').value = '';
}

function store(id_sta) {
    var val = document.getElementById('myvariables').value;
    document.getElementById('myvariables').value = val + id_sta + "' ";
}

function newnpc() {
    const row = document.getElementById('npc_tbl_id').insertRow();
    const ls = getall('npc');
    const nr = ls.length;
    const id_sta = "'npc_" + nr.toString();
    store(id_sta);

    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><input type='text' id=" + id_sta + "_name' name=" + id_sta + "_name'></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><select id=" + id_sta + "_flag' name=" + id_sta + "_flag'>\n" + dropdown(getall('fla')) + "</select></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><textarea id=" + id_sta + "_desc' name=" + id_sta + "_desc'></textarea></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><select id=" + id_sta + "_secr' name=" + id_sta + "_secr' multiple>\n" + dropdown(getall('sec')) + "</select></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><input type='number' step='0.1' id=" + id_sta + "_appe' name=" + id_sta + "_appe'></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><textarea id=" + id_sta + "_item' name=" + id_sta + "_item'></textarea></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><textarea id=" + id_sta + "_skil' name=" + id_sta + "_skil'></textarea></td></tr>";
    /*
    newcell(newRow, 0, id_sta + "_name'");
    newcell(newRow, 2, id_sta + "_flag'", dropdown(getall('fla')));
    newcell(newRow, 1, id_sta + "_desc'");
    newcell(newRow, 2, id_sta + "_secr", dropdown(getall('sec')), "[]' multiple");
    newcell(newRow, 0, id_sta + "_appe'");
     */
}

function newloc() {
    const row = document.getElementById('loc_tbl_id').insertRow();
    const ls = getall('loc');
    const nr = ls.length;
    const id_sta = "'loc_" + nr.toString();
    store(id_sta);

    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><input type='text' id=" + id_sta + "_name' name=" + id_sta + "_name'></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><select id=" + id_sta + "_flag' name=" + id_sta + "_flag'>\n" + dropdown(getall('fla')) + "</select></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><textarea id=" + id_sta + "_desc' name=" + id_sta + "_desc'></textarea></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><select id=" + id_sta + "_secr' name=" + id_sta + "_secr' multiple>\n" + dropdown(getall('sec')) + "</select></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><textarea id=" + id_sta + "_item' name=" + id_sta + "_item'></textarea></td></tr>";
    /*
    newcell(newRow, 0, id_sta + "_name'");
    newcell(newRow, 2, id_sta + "_flag'", dropdown(getall('fla')));
    newcell(newRow, 1, id_sta + "_desc'");
    newcell(newRow, 2, id_sta + "_secr", dropdown(getall('sec')), "[]' multiple");
    */
}

function newsec() {
    const row = document.getElementById('sec_tbl_id').insertRow();
    const ls = getall('sec');
    const nr = ls.length;
    const id_sta = "'sec_" + nr.toString();
    store(id_sta);

    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><input type='text' id=" + id_sta + "_name' name=" + id_sta + "_name'></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><select id=" + id_sta + "_flag' name=" + id_sta + "_flag'>\n" + dropdown(getall('fla')) + "</select></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><textarea id=" + id_sta + "_prom' name=" + id_sta + "_prom'></textarea></td></tr>";

    /*
    newcell(newRow, 0, id_sta + "_name'");
    newcell(newRow, 2, id_sta + "_flag'", dropdown(getall('fla')));
    newcell(newRow, 1, id_sta + "_prom'");
     */
}

function newfla() {
    const row = document.getElementById('fla_tbl_id').insertRow();
    const ls = getall('fla');
    const nr = ls.length;
    const id_sta = "'fla_" + nr.toString();
    store(id_sta);

    var full_ls = [];
    full_ls = full_ls.concat(ls, getall('npc'), getall('loc'), getall('sec'));

    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><input type='text' id=" + id_sta + "_name' name=" + id_sta + "_name'></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><select id=" + id_sta + "_cond' name=" + id_sta + "_cond' multiple>\n" + dropdown(full_ls) + "</select></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><textarea id=" + id_sta + "_ccon' name=" + id_sta + "_desc'></textarea></td></tr>";

    /*
    newcell(newRow, 0, id_sta + "_name'");
    var full_ls = [];
    full_ls = full_ls.concat(ls, getall('npc'), getall('loc'), getall('sec'));
    newcell(newRow, 2, id_sta + "_cond", dropdown(full_ls), "[]' multiple");
    newcell(newRow, 1, id_sta + "_ccon'");
     */
}

function newtri() {
    const row = document.getElementById('tri_tbl_id').insertRow();
    const ls = getall('tri');
    const nr = ls.length;
    const id_sta = "'tri_" + nr.toString();
    store(id_sta);

    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><input type='text' id=" + id_sta + "_name' name=" + id_sta + "_name'></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><select id=" + id_sta + "_flag' name=" + id_sta + "_flag'>\n" + dropdown(getall('fla')) + "</select></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><select id=" + id_sta + "_call' name=" + id_sta + "_call'>\n" + dropdown(getall('fla')) + "</select></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><input type='text' id=" + id_sta + "_func' name=" + id_sta + "_func'></td></tr>";
}

function stasta() {
    const row = document.getElementById('sta_tbl_id').insertRow();
    const id_sta = "'sta_"


    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><select id=" + id_sta + "loca' name=" + id_sta + "loca'>\n" + dropdown(getall('loc')) + "</select></td></tr>";
    var newCell = row.insertCell();
    newCell.innerHTML="<tr><td><select id=" + id_sta + "npcs' name=" + id_sta + "npcs' multiple>\n" + dropdown(getall('npc')) + "</select></td></tr>";
}
