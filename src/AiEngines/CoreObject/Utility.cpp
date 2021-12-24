#include "Utility.hpp"

vector<string> loadObjectNames(const string &filename) {
    ifstream file(filename);
    vector<string> fileLines;
    if (!file.is_open()) {
        return fileLines;
    }
    string line;
    while (getline(file, line)) {
        fileLines.push_back(line);
        //std::cout << line << "\n";
    }
    std::cout << "[INFO] Object names were loaded successfully! \n";
    return fileLines;
}

void bbox2regions(vector<bbox_t>& bboxs, regions_t& regions, vector<string>& objNames) {
    for (const bbox_t& bbox : bboxs) {
        if (bbox.w * bbox.h > 300) {
            regions.emplace_back(Rect(bbox.x, bbox.y, bbox.w, bbox.h), objNames[bbox.obj_id], bbox.prob);
        }
	}
}

unsigned int combineObject(unsigned int obj_id) {
    unsigned int combine = 0;
    switch (obj_id) {
        case 1:
        case 2:
            combine = XE2BANH;
            break;
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
            combine = XE4BANH;
            break;
        default:
            combine =0;
            break;
    }
    return combine;
}

int convertType2Int(vector<string>& objNames, string type) {
    for (int index = 0; index <objNames.size(); index++) {
        if (type == objNames[index]) return index;
    }
    return 0;
}